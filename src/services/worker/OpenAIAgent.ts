/**
 * OpenAIAgent: OpenAI-based observation extraction
 *
 * Alternative to SDKAgent that uses OpenAI's API directly
 * for accessing GPT models.
 *
 * Responsibility:
 * - Call OpenAI REST API for observation extraction
 * - Parse XML responses (same format as Claude/Gemini/OpenRouter)
 * - Sync to database and Chroma
 * - Support dynamic model selection
 */

import { buildContinuationPrompt, buildInitPrompt, buildObservationPrompt, buildSummaryPrompt } from '../../sdk/prompts.js';
import { getCredential } from '../../shared/EnvManager.js';
import { SettingsDefaultsManager } from '../../shared/SettingsDefaultsManager.js';
import { USER_SETTINGS_PATH } from '../../shared/paths.js';
import { logger } from '../../utils/logger.js';
import { ModeManager } from '../domain/ModeManager.js';
import type { ActiveSession, ConversationMessage } from '../worker-types.js';
import { DatabaseManager } from './DatabaseManager.js';
import { SessionManager } from './SessionManager.js';
import {
  isAbortError,
  processAgentResponse,
  shouldFallbackToClaude,
  type FallbackAgent,
  type WorkerRef
} from './agents/index.js';

// Default OpenAI API base URL (can be overridden via CLAUDE_MEM_OPENAI_BASE_URL for Azure/proxies)
const DEFAULT_OPENAI_BASE_URL = 'https://api.openai.com/v1';

// Context window management constants (defaults, overridable via settings)
const DEFAULT_MAX_CONTEXT_MESSAGES = 20;
const DEFAULT_MAX_ESTIMATED_TOKENS = 100000;
const CHARS_PER_TOKEN_ESTIMATE = 4;

// OpenAI-compatible message format
interface OpenAIMessage {
  role: 'user' | 'assistant' | 'system';
  content: string;
}

interface OpenAIResponse {
  choices?: Array<{
    message?: {
      role?: string;
      content?: string;
    };
    finish_reason?: string;
  }>;
  usage?: {
    prompt_tokens?: number;
    completion_tokens?: number;
    total_tokens?: number;
  };
  error?: {
    message?: string;
    code?: string;
    type?: string;
  };
}

export class OpenAIAgent {
  private dbManager: DatabaseManager;
  private sessionManager: SessionManager;
  private fallbackAgent: FallbackAgent | null = null;

  constructor(dbManager: DatabaseManager, sessionManager: SessionManager) {
    this.dbManager = dbManager;
    this.sessionManager = sessionManager;
  }

  /**
   * Set the fallback agent (Claude SDK) for when OpenAI API fails
   * Must be set after construction to avoid circular dependency
   */
  setFallbackAgent(agent: FallbackAgent): void {
    this.fallbackAgent = agent;
  }

  /**
   * Start OpenAI agent for a session
   * Uses multi-turn conversation to maintain context across messages
   */
  async startSession(session: ActiveSession, worker?: WorkerRef): Promise<void> {
    try {
      // Get OpenAI configuration
      const { apiKey, model, baseUrl } = this.getOpenAIConfig();

      if (!apiKey) {
        throw new Error('OpenAI API key not configured. Set CLAUDE_MEM_OPENAI_API_KEY in settings or OPENAI_API_KEY environment variable.');
      }

      // Generate synthetic memorySessionId (OpenAI is stateless, doesn't return session IDs)
      if (!session.memorySessionId) {
        const syntheticMemorySessionId = `openai-${session.contentSessionId}-${Date.now()}`;
        session.memorySessionId = syntheticMemorySessionId;
        this.dbManager.getSessionStore().updateMemorySessionId(session.sessionDbId, syntheticMemorySessionId);
        logger.info('SESSION', `MEMORY_ID_GENERATED | sessionDbId=${session.sessionDbId} | provider=OpenAI`);
      }

      // Load active mode
      const mode = ModeManager.getInstance().getActiveMode();

      // Build initial prompt
      const initPrompt = session.lastPromptNumber === 1
        ? buildInitPrompt(session.project, session.contentSessionId, session.userPrompt, mode)
        : buildContinuationPrompt(session.userPrompt, session.lastPromptNumber, session.contentSessionId, mode);

      // Add to conversation history and query OpenAI with full context
      session.conversationHistory.push({ role: 'user', content: initPrompt });
      const initResponse = await this.queryOpenAIMultiTurn(session.conversationHistory, apiKey, model, baseUrl);

      if (initResponse.content) {
        const tokensUsed = initResponse.tokensUsed || 0;
        session.cumulativeInputTokens += Math.floor(tokensUsed * 0.7);
        session.cumulativeOutputTokens += Math.floor(tokensUsed * 0.3);

        await processAgentResponse(
          initResponse.content,
          session,
          this.dbManager,
          this.sessionManager,
          worker,
          tokensUsed,
          null,
          'OpenAI',
          undefined,
          model
        );
      } else {
        logger.error('SDK', 'Empty OpenAI init response - session may lack context', {
          sessionId: session.sessionDbId,
          model
        });
      }

      // Track lastCwd from messages for CLAUDE.md generation
      let lastCwd: string | undefined;

      // Process pending messages
      for await (const message of this.sessionManager.getMessageIterator(session.sessionDbId)) {
        session.processingMessageIds.push(message._persistentId);

        if (message.cwd) {
          lastCwd = message.cwd;
        }
        const originalTimestamp = session.earliestPendingTimestamp;

        if (message.type === 'observation') {
          if (message.prompt_number !== undefined) {
            session.lastPromptNumber = message.prompt_number;
          }

          if (!session.memorySessionId) {
            throw new Error('Cannot process observations: memorySessionId not yet captured. This session may need to be reinitialized.');
          }

          const obsPrompt = buildObservationPrompt({
            id: 0,
            tool_name: message.tool_name!,
            tool_input: JSON.stringify(message.tool_input),
            tool_output: JSON.stringify(message.tool_response),
            created_at_epoch: originalTimestamp ?? Date.now(),
            cwd: message.cwd
          });

          session.conversationHistory.push({ role: 'user', content: obsPrompt });
          const obsResponse = await this.queryOpenAIMultiTurn(session.conversationHistory, apiKey, model, baseUrl);

          let tokensUsed = 0;
          if (obsResponse.content) {
            tokensUsed = obsResponse.tokensUsed || 0;
            session.cumulativeInputTokens += Math.floor(tokensUsed * 0.7);
            session.cumulativeOutputTokens += Math.floor(tokensUsed * 0.3);
          }

          await processAgentResponse(
            obsResponse.content || '',
            session,
            this.dbManager,
            this.sessionManager,
            worker,
            tokensUsed,
            originalTimestamp,
            'OpenAI',
            lastCwd,
            model
          );

        } else if (message.type === 'summarize') {
          if (!session.memorySessionId) {
            throw new Error('Cannot process summary: memorySessionId not yet captured. This session may need to be reinitialized.');
          }

          const summaryPrompt = buildSummaryPrompt({
            id: session.sessionDbId,
            memory_session_id: session.memorySessionId,
            project: session.project,
            user_prompt: session.userPrompt,
            last_assistant_message: message.last_assistant_message || ''
          }, mode);

          session.conversationHistory.push({ role: 'user', content: summaryPrompt });
          const summaryResponse = await this.queryOpenAIMultiTurn(session.conversationHistory, apiKey, model, baseUrl);

          let tokensUsed = 0;
          if (summaryResponse.content) {
            tokensUsed = summaryResponse.tokensUsed || 0;
            session.cumulativeInputTokens += Math.floor(tokensUsed * 0.7);
            session.cumulativeOutputTokens += Math.floor(tokensUsed * 0.3);
          }

          await processAgentResponse(
            summaryResponse.content || '',
            session,
            this.dbManager,
            this.sessionManager,
            worker,
            tokensUsed,
            originalTimestamp,
            'OpenAI',
            lastCwd,
            model
          );
        }
      }

      const sessionDuration = Date.now() - session.startTime;
      logger.success('SDK', 'OpenAI agent completed', {
        sessionId: session.sessionDbId,
        duration: `${(sessionDuration / 1000).toFixed(1)}s`,
        historyLength: session.conversationHistory.length,
        model
      });

    } catch (error: unknown) {
      if (isAbortError(error)) {
        logger.warn('SDK', 'OpenAI agent aborted', { sessionId: session.sessionDbId });
        throw error;
      }

      if (shouldFallbackToClaude(error) && this.fallbackAgent) {
        logger.warn('SDK', 'OpenAI API failed, falling back to Claude SDK', {
          sessionDbId: session.sessionDbId,
          error: error instanceof Error ? error.message : String(error),
          historyLength: session.conversationHistory.length
        });
        return this.fallbackAgent.startSession(session, worker);
      }

      logger.failure('SDK', 'OpenAI agent error', { sessionDbId: session.sessionDbId }, error as Error);
      throw error;
    }
  }

  /**
   * Estimate token count from text (conservative estimate)
   */
  private estimateTokens(text: string): number {
    return Math.ceil(text.length / CHARS_PER_TOKEN_ESTIMATE);
  }

  /**
   * Truncate conversation history to prevent runaway context costs
   */
  private truncateHistory(history: ConversationMessage[]): ConversationMessage[] {
    const settings = SettingsDefaultsManager.loadFromFile(USER_SETTINGS_PATH);

    const MAX_CONTEXT_MESSAGES = parseInt(settings.CLAUDE_MEM_OPENAI_MAX_CONTEXT_MESSAGES) || DEFAULT_MAX_CONTEXT_MESSAGES;
    const MAX_ESTIMATED_TOKENS = parseInt(settings.CLAUDE_MEM_OPENAI_MAX_TOKENS) || DEFAULT_MAX_ESTIMATED_TOKENS;

    if (history.length <= MAX_CONTEXT_MESSAGES) {
      const totalTokens = history.reduce((sum, m) => sum + this.estimateTokens(m.content), 0);
      if (totalTokens <= MAX_ESTIMATED_TOKENS) {
        return history;
      }
    }

    const truncated: ConversationMessage[] = [];
    let tokenCount = 0;

    for (let i = history.length - 1; i >= 0; i--) {
      const msg = history[i];
      const msgTokens = this.estimateTokens(msg.content);

      if (truncated.length >= MAX_CONTEXT_MESSAGES || tokenCount + msgTokens > MAX_ESTIMATED_TOKENS) {
        logger.warn('SDK', 'Context window truncated to prevent runaway costs', {
          originalMessages: history.length,
          keptMessages: truncated.length,
          droppedMessages: i + 1,
          estimatedTokens: tokenCount,
          tokenLimit: MAX_ESTIMATED_TOKENS
        });
        break;
      }

      truncated.unshift(msg);
      tokenCount += msgTokens;
    }

    return truncated;
  }

  /**
   * Convert shared ConversationMessage array to OpenAI message format
   */
  private conversationToOpenAIMessages(history: ConversationMessage[]): OpenAIMessage[] {
    return history.map(msg => ({
      role: msg.role === 'assistant' ? 'assistant' : 'user',
      content: msg.content
    }));
  }

  /**
   * Query OpenAI via REST API with full conversation history (multi-turn)
   */
  private async queryOpenAIMultiTurn(
    history: ConversationMessage[],
    apiKey: string,
    model: string,
    baseUrl: string
  ): Promise<{ content: string; tokensUsed?: number }> {
    const truncatedHistory = this.truncateHistory(history);
    const messages = this.conversationToOpenAIMessages(truncatedHistory);
    const totalChars = truncatedHistory.reduce((sum, m) => sum + m.content.length, 0);
    const estimatedTokens = this.estimateTokens(truncatedHistory.map(m => m.content).join(''));

    logger.debug('SDK', `Querying OpenAI multi-turn (${model})`, {
      turns: truncatedHistory.length,
      totalChars,
      estimatedTokens
    });

    const response = await fetch(`${baseUrl}/chat/completions`, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${apiKey}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        model,
        messages,
        temperature: 0.3,
      }),
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`OpenAI API error: ${response.status} - ${errorText}`);
    }

    const data = await response.json() as OpenAIResponse;

    if (data.error) {
      throw new Error(`OpenAI API error: ${data.error.code} - ${data.error.message}`);
    }

    if (!data.choices?.[0]?.message?.content) {
      logger.error('SDK', 'Empty response from OpenAI');
      return { content: '' };
    }

    const content = data.choices[0].message.content;
    const tokensUsed = data.usage?.total_tokens;

    if (tokensUsed) {
      const inputTokens = data.usage?.prompt_tokens || 0;
      const outputTokens = data.usage?.completion_tokens || 0;
      // Rough cost estimate (gpt-4o-mini pricing as baseline)
      const estimatedCost = (inputTokens / 1000000 * 0.15) + (outputTokens / 1000000 * 0.6);

      logger.info('SDK', 'OpenAI API usage', {
        model,
        inputTokens,
        outputTokens,
        totalTokens: tokensUsed,
        estimatedCostUSD: estimatedCost.toFixed(4),
        messagesInContext: truncatedHistory.length
      });

      if (tokensUsed > 50000) {
        logger.warn('SDK', 'High token usage detected - consider reducing context', {
          totalTokens: tokensUsed,
          estimatedCost: estimatedCost.toFixed(4)
        });
      }
    }

    return { content, tokensUsed };
  }

  /**
   * Get OpenAI configuration from settings or environment
   */
  private getOpenAIConfig(): { apiKey: string; model: string; baseUrl: string } {
    const settings = SettingsDefaultsManager.loadFromFile(USER_SETTINGS_PATH);

    const apiKey = settings.CLAUDE_MEM_OPENAI_API_KEY || getCredential('OPENAI_API_KEY') || '';
    const model = settings.CLAUDE_MEM_OPENAI_MODEL || 'gpt-4o-mini';
    const baseUrl = settings.CLAUDE_MEM_OPENAI_BASE_URL || DEFAULT_OPENAI_BASE_URL;

    return { apiKey, model, baseUrl };
  }
}

/**
 * Check if OpenAI is available (has API key configured)
 */
export function isOpenAIAvailable(): boolean {
  const settings = SettingsDefaultsManager.loadFromFile(USER_SETTINGS_PATH);
  return !!(settings.CLAUDE_MEM_OPENAI_API_KEY || getCredential('OPENAI_API_KEY'));
}

/**
 * Check if OpenAI is the selected provider
 */
export function isOpenAISelected(): boolean {
  const settings = SettingsDefaultsManager.loadFromFile(USER_SETTINGS_PATH);
  return settings.CLAUDE_MEM_PROVIDER === 'openai';
}
