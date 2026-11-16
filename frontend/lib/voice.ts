/** Browser speech helpers for Thai and English commands. */

export type VoiceLang = "th-TH" | "en-US";

type SpeechRecognitionResultLike = {
  isFinal?: boolean;
  length: number;
  [index: number]: { transcript: string; confidence?: number };
};

type SpeechRecognitionEventLike = {
  resultIndex: number;
  results: ArrayLike<SpeechRecognitionResultLike> & { length: number };
};

type SpeechRecognitionLike = {
  lang: string;
  continuous: boolean;
  interimResults: boolean;
  maxAlternatives: number;
  onresult: ((event: SpeechRecognitionEventLike) => void) | null;
  onerror: ((event: { error?: string }) => void) | null;
  onend: (() => void) | null;
  start: () => void;
  stop: () => void;
  abort: () => void;
};

type SpeechWindow = Window & {
  SpeechRecognition?: new () => SpeechRecognitionLike;
  webkitSpeechRecognition?: new () => SpeechRecognitionLike;
};

export function hasThai(text: string) {
  return /[\u0E00-\u0E7F]/.test(text);
}

export function speechSupported() {
  if (typeof window === "undefined") return false;
  const speech = window as SpeechWindow;
  return Boolean(speech.SpeechRecognition || speech.webkitSpeechRecognition);
}

export function speak(text: string, lang?: VoiceLang): Promise<void> {
  return new Promise((resolve) => {
    if (typeof window === "undefined" || !window.speechSynthesis) {
      resolve();
      return;
    }
    window.speechSynthesis.cancel();
    const line = text.trim();
    if (!line) {
      resolve();
      return;
    }
    const speech = new SpeechSynthesisUtterance(line);
    speech.lang = lang || (hasThai(line) ? "th-TH" : "en-US");
    speech.rate = lang === "th-TH" ? 0.95 : 1;
    speech.pitch = 1;
    speech.onend = () => resolve();
    speech.onerror = () => resolve();
    // Prefer a matching system voice when macOS has one installed.
    const voices = window.speechSynthesis.getVoices();
    const wanted = speech.lang.slice(0, 2).toLowerCase();
    const match = voices.find((voice) => voice.lang.toLowerCase().startsWith(wanted));
    if (match) speech.voice = match;
    window.speechSynthesis.speak(speech);
  });
}

function pickTranscript(result: SpeechRecognitionResultLike, lang: VoiceLang) {
  const options: Array<{ text: string; confidence: number }> = [];
  for (let index = 0; index < result.length; index += 1) {
    const item = result[index];
    const text = String(item?.transcript || "").trim();
    if (!text) continue;
    options.push({ text, confidence: Number(item?.confidence ?? 0) });
  }
  if (!options.length) return "";
  if (lang === "th-TH") {
    const thai = options.filter((item) => hasThai(item.text));
    if (thai.length) return thai.sort((a, b) => b.confidence - a.confidence)[0].text;
  }
  return options.sort((a, b) => b.confidence - a.confidence)[0].text;
}

export function listen(
  lang: VoiceLang,
  timeoutMs = 10000,
  endSilenceMs = 900,
): { promise: Promise<string>; stop: () => void } {
  let recognition: SpeechRecognitionLike | null = null;
  let timer = 0;
  let endTimer = 0;
  let settled = false;
  const finals: string[] = [];

  const clearTimers = () => {
    if (timer) window.clearTimeout(timer);
    if (endTimer) window.clearTimeout(endTimer);
    timer = 0;
    endTimer = 0;
  };

  const stop = () => {
    clearTimers();
    try {
      recognition?.stop();
    } catch {
      try {
        recognition?.abort();
      } catch {
        /* ignore */
      }
    }
  };

  const finish = (resolve: (value: string) => void, value: string) => {
    if (settled) return;
    settled = true;
    clearTimers();
    resolve(value.trim());
  };

  const promise = new Promise<string>((resolve, reject) => {
    if (typeof window === "undefined") {
      reject(new Error("Speech is not available."));
      return;
    }
    const speech = window as SpeechWindow;
    const Ctor = speech.SpeechRecognition || speech.webkitSpeechRecognition;
    if (!Ctor) {
      reject(new Error("This browser cannot listen. Use Chrome for voice commands."));
      return;
    }

    const armEndOfSpeech = () => {
      if (settled || !finals.length) return;
      if (endTimer) window.clearTimeout(endTimer);
      // Stop soon after the last final phrase so recording does not hang open.
      endTimer = window.setTimeout(() => stop(), endSilenceMs);
    };

    recognition = new Ctor();
    recognition.lang = lang;
    // continuous + short end silence: keep short Thai pauses, stop when speech ends.
    recognition.continuous = true;
    recognition.interimResults = true;
    recognition.maxAlternatives = 5;
    recognition.onresult = (event) => {
      let gotFinal = false;
      for (let index = event.resultIndex; index < event.results.length; index += 1) {
        const result = event.results[index];
        if (!result?.isFinal) {
          // Still speaking — wait for the next pause.
          if (endTimer) window.clearTimeout(endTimer);
          endTimer = 0;
          continue;
        }
        const text = pickTranscript(result, lang);
        if (text) {
          finals.push(text);
          gotFinal = true;
        }
      }
      if (gotFinal) armEndOfSpeech();
    };
    recognition.onerror = (event) => {
      if (settled) return;
      const code = event.error || "";
      if (code === "no-speech" || code === "aborted") {
        finish(resolve, finals.join(" "));
        return;
      }
      settled = true;
      clearTimers();
      reject(new Error(code || "Speech recognition failed."));
    };
    recognition.onend = () => {
      finish(resolve, finals.join(" "));
    };
    timer = window.setTimeout(() => stop(), timeoutMs);
    try {
      recognition.start();
    } catch (error) {
      if (!settled) {
        settled = true;
        clearTimers();
        reject(error instanceof Error ? error : new Error("Could not start the microphone."));
      }
    }
  });

  return { promise, stop };
}

/** Thai mode must hear Thai script. English-looking STT is usually a bad transcription. */
export function voiceLooksWrong(lang: VoiceLang, heard: string) {
  const text = heard.trim();
  if (!text) return true;
  if (lang === "th-TH" && !hasThai(text)) return true;
  return false;
}
