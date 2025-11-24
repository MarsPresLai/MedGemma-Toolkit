const normalizeBaseUrl = (value?: string) => {
  if (!value) {
    return '/api';
  }
  return value.endsWith('/') ? value.slice(0, -1) : value;
};

const API_BASE_URL = normalizeBaseUrl(import.meta.env.VITE_API_BASE_URL as string | undefined);
const ANALYZE_ENDPOINT = `${API_BASE_URL}/analyze/`;

export interface StreamDonePayload {
  vlm_output?: string;
  llm_report?: string;
}

export interface AnalyzeStreamHandlers {
  onStatus?: (state: string) => void;
  onVlmToken?: (token: string) => void;
  onLlmToken?: (token: string) => void;
  onDone?: (payload: StreamDonePayload) => void;
  onError?: (message: string) => void;
}

const parseEventPayload = (raw: string): Record<string, any> | null => {
  const cleaned = raw.replace(/^data:\s*/, '').trim();
  if (!cleaned) {
    return null;
  }

  try {
    return JSON.parse(cleaned);
  } catch (err) {
    console.error('Failed to parse SSE payload', err, cleaned);
    return null;
  }
};

export const analyzeCaseStream = async (
  prompt: string,
  image: File | null,
  handlers: AnalyzeStreamHandlers,
): Promise<void> => {
  const formData = new FormData();
  formData.append('prompt', prompt);
  if (image) {
    formData.append('image', image);
  }

  const response = await fetch(ANALYZE_ENDPOINT, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok || !response.body) {
    const detail = await response
      .json()
      .catch(() => ({ detail: response.statusText }));
    throw new Error(detail?.detail || 'Request failed');
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  while (true) {
    const { value, done } = await reader.read();
    if (done) {
      break;
    }

    buffer += decoder.decode(value, { stream: true });

    let boundary = buffer.search(/\r?\n\r?\n/);
    while (boundary !== -1) {
      const chunk = buffer.slice(0, boundary);
      buffer = buffer.slice(boundary + (buffer[boundary] === '\r' ? 4 : 2));

      const lines = chunk.split(/\r?\n/).map((line) => line.trim());
      const dataLine = lines.find((line) => line.startsWith('data:'));
      if (!dataLine || dataLine.startsWith(':')) {
        boundary = buffer.search(/\r?\n\r?\n/);
        continue;
      }

      const payload = parseEventPayload(dataLine);
      if (payload?.event === 'status' && payload.state) {
        handlers.onStatus?.(payload.state);
      } else if (payload?.event === 'vlm_token' && payload.token) {
        handlers.onVlmToken?.(payload.token);
      } else if (payload?.event === 'llm_token' && payload.token) {
        handlers.onLlmToken?.(payload.token);
      } else if (payload?.event === 'done') {
        handlers.onDone?.({
          vlm_output: payload.vlm_output,
          llm_report: payload.llm_report,
        });
      } else if (payload?.event === 'error') {
        handlers.onError?.(payload.message || 'Stream error');
      }

      boundary = buffer.search(/\r?\n\r?\n/);
    }
  }
};
