"use client"

import type React from "react"
import { VoiceAssistantLiveKit } from "./VoiceAssistantLiveKit";

import ReactMarkdown from "react-markdown"
import remarkGfm from "remark-gfm"



import { useState, useRef, useEffect } from "react"
import { Card } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import {
  MessageCircle,
  X,
  Send,
  Bot,
  User,
  Mic,
  Volume2,
  ChevronDown,
  ChevronUp,
} from "lucide-react"

interface Message {
  id: string
  text: string
  sender: "user" | "bot"
  timestamp: Date
  sources?: { title: string; url: string }[]
}

export function ChatBot() {
  const [isOpen, setIsOpen] = useState(false)
  const [voiceMode, setVoiceMode] = useState(false)
  const [showTooltip, setShowTooltip] = useState(true)
  const [tooltipDismissed, setTooltipDismissed] = useState(false)
  const [isClosing, setIsClosing] = useState(false)
  const [isTyping, setIsTyping] = useState(false)
  const [voiceAssistantEnabled, setVoiceAssistantEnabled] = useState(false)
  const [livekitToken, setLivekitToken] = useState<string | null>(null);
  





  // Mic state
  const [isRecording, setIsRecording] = useState(false)
  const mediaRecorderRef = useRef<MediaRecorder | null>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const audioChunksRef = useRef<Blob[]>([])

  // TTS playback
  const audioRef = useRef<HTMLAudioElement | null>(null)
  const lastAudioUrlRef = useRef<string | null>(null)
  const ttsAbortRef = useRef<AbortController | null>(null)
  const mediaSourceRef = useRef<MediaSource | null>(null)
  const voiceLoopActiveRef = useRef(false)
  

  


  useEffect(() => {
  audioRef.current = new Audio()
  audioRef.current.preload = "auto"

  return () => {
    audioRef.current?.pause()
    audioRef.current = null
  }
}, [])



  const [expandedSources, setExpandedSources] = useState<{ [key: string]: boolean }>({})
  const [messages, setMessages] = useState<Message[]>([
    {
      id: "1",
      text: "Hi! Welcome to LMKR. How can I assist you today?",
      sender: "bot",
      timestamp: new Date(),
    },
  ])

  const [inputValue, setInputValue] = useState("")
  const [threadId, setThreadId] = useState<string | null>(null)
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const abortControllerRef = useRef<AbortController | null>(null)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  useEffect(() => {
    // Cleanup on unmount
    return () => {
      if (abortControllerRef.current) {
        abortControllerRef.current.abort()
      }
      // Stop any playing TTS
      if (audioRef.current) {
        audioRef.current.pause()
      }
      // Revoke any blob URL
      if (lastAudioUrlRef.current) {
        URL.revokeObjectURL(lastAudioUrlRef.current)
        lastAudioUrlRef.current = null
      }
      // Stop mic stream if active
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((t) => t.stop())
        streamRef.current = null
      }
    }
  }, [])

  useEffect(() => {
    if (tooltipDismissed) return

    const timer = setTimeout(() => {
      setShowTooltip(false)
      setTooltipDismissed(true)
    }, 10000)

    return () => clearTimeout(timer)
  }, [tooltipDismissed])

  const stopTtsPlayback = () => {
  // Abort streaming fetch
  if (ttsAbortRef.current) {
    ttsAbortRef.current.abort()
    ttsAbortRef.current = null
  }

  // Stop audio immediately
  if (audioRef.current) {
    audioRef.current.pause()
    audioRef.current.currentTime = 0
    audioRef.current.src = ""
  }

  // Close MediaSource cleanly
  if (mediaSourceRef.current) {
    try {
      if (mediaSourceRef.current.readyState === "open") {
        mediaSourceRef.current.endOfStream()
      }
    } catch {}
    mediaSourceRef.current = null
  }
}





  const speakText = async (text: string) => {
  if (!text || !text.trim()) return

  try {
    // 🔴 Stop anything already speaking
    stopTtsPlayback()

    const apiUrl =
      process.env.NEXT_PUBLIC_FLASK_API_URL || "http://localhost:8000"

    if (!audioRef.current) {
      audioRef.current = new Audio()
      audioRef.current.preload = "auto"
    }

    const audio = audioRef.current
    const mediaSource = new MediaSource()
    mediaSourceRef.current = mediaSource

    audio.src = URL.createObjectURL(mediaSource)

    // IMPORTANT: do NOT await play()
    audio.play().catch((err) => {
      if (err.name !== "AbortError") {
        console.error("Audio play failed:", err)
      }
    })

    // Abort controller for this TTS stream
    const abortController = new AbortController()
    ttsAbortRef.current = abortController

    mediaSource.addEventListener("sourceopen", async () => {
      let sourceBuffer: SourceBuffer

      try {
        sourceBuffer = mediaSource.addSourceBuffer("audio/mpeg")
      } catch (e) {
        console.error("Failed to create SourceBuffer:", e)
        return
      }

      const res = await fetch(`${apiUrl}/speak_stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text, voice: "alloy" }),
        signal: abortController.signal,
      })

      if (!res.body) return

      const reader = res.body.getReader()

      try {
        while (true) {
          const { value, done } = await reader.read()
          if (done) break

          if (sourceBuffer.updating) {
            await new Promise((resolve) =>
              sourceBuffer.addEventListener("updateend", resolve, { once: true })
            )
          }

          sourceBuffer.appendBuffer(value)
        }

        if (mediaSource.readyState === "open") {
          mediaSource.endOfStream()
        }
      } catch (err: any) {
        if (err.name !== "AbortError") {
          console.error("TTS stream error:", err)
        }
      }
    })

    // 🔁 CHATGPT-STYLE VOICE LOOP
    audio.onended = async () => {
      ttsAbortRef.current = null

      if (!voiceLoopActiveRef.current) return

      // Small delay so mic doesn't clip user speech
      setTimeout(async () => {
        if (!voiceLoopActiveRef.current) return

        try {
          await startRecordingBase(true)
        } catch (e) {
          console.error("Failed to restart voice assistant mic:", e)
        }
      }, 500)
    }
  } catch (err) {
    console.error("TTS playback failed:", err)
  }
}





  const handleSend = async () => {
  if (!inputValue.trim()) return

  const question = inputValue.trim()

  // Stop any currently playing TTS when user sends a new message
  stopTtsPlayback()

  // Add user message
  const userMessage: Message = {
    id: Date.now().toString(),
    text: question,
    sender: "user",
    timestamp: new Date(),
  }
  setMessages((prev) => [...prev, userMessage])
  setInputValue("")
  setIsTyping(true)

  // Placeholder bot message for streaming
  const botMessageId = (Date.now() + 1).toString()
  const botMessage: Message = {
    id: botMessageId,
    text: "",
    sender: "bot",
    timestamp: new Date(),
    sources: [],
  }
  setMessages((prev) => [...prev, botMessage])

  // Abort previous request
  if (abortControllerRef.current) {
    abortControllerRef.current.abort()
  }
  abortControllerRef.current = new AbortController()

  try {
    const apiUrl = process.env.NEXT_PUBLIC_FLASK_API_URL || "http://localhost:8000"

    const response = await fetch(`${apiUrl}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question,
        thread_id: threadId,
        stream: true,
      }),
      signal: abortControllerRef.current.signal,
    })

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`)
    }

    const reader = response.body?.getReader()
    const decoder = new TextDecoder()

    let buffer = ""
    let accumulatedText = ""
    let sources: string[] = []
    let receivedThreadId: string | null = null

    if (!reader) throw new Error("No response body")

    while (true) {
      const { done, value } = await reader.read()
      if (done) break

      buffer += decoder.decode(value, { stream: true })
      const lines = buffer.split("\n")
      buffer = lines.pop() || ""

      for (const line of lines) {
        if (!line.startsWith("data: ")) continue

        try {
          const data = JSON.parse(line.slice(6))

          if (data.type === "chunk") {
            accumulatedText += data.data
            setMessages((prev) =>
              prev.map((msg) =>
                msg.id === botMessageId ? { ...msg, text: accumulatedText } : msg
              )
            )
            scrollToBottom()
          }

          else if (data.type === "sources") {
            sources = data.data || []
          }

          else if (data.type === "thread_id") {
            receivedThreadId = data.data
            if (receivedThreadId) setThreadId(receivedThreadId)
          }

          else if (data.type === "error") {
            const errorData = data.data
            setMessages((prev) =>
              prev.map((msg) =>
                msg.id === botMessageId
                  ? {
                      ...msg,
                      text: errorData.answer || "An error occurred.",
                      sources:
                        errorData.sources?.map((s: string) => ({
                          title: s,
                          url: s,
                        })) || [],
                    }
                  : msg
              )
            )
            if (errorData.thread_id) setThreadId(errorData.thread_id)
          }

          else if (data.type === "done") {
            // Finalize message sources
            setMessages((prev) =>
              prev.map((msg) =>
                msg.id === botMessageId
                  ? {
                      ...msg,
                      sources: sources.map((s) => ({
                        title: s,
                        url: s,
                      })),
                    }
                  : msg
              )
            )

            // 🔊 AUTO SPEAK ONLY IN VOICE MODE
            if (!voiceAssistantEnabled && accumulatedText.trim()) {
            await speakText(accumulatedText)
          }      

            
          }
        } catch (e) {
          console.error("Error parsing SSE data:", e)
        }
      }
    }

    setIsTyping(false)
  } catch (error: any) {
    if (error?.name === "AbortError") return

    console.error("Error calling chat API:", error)
    setIsTyping(false)

    setMessages((prev) =>
      prev.map((msg) =>
        msg.id === botMessageId
          ? { ...msg, text: "Sorry, I encountered an error. Please try again." }
          : msg
      )
    )
  }
}


  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  const handleClose = () => {
    setIsClosing(true)
    setTimeout(() => {
      setIsOpen(false)
      setIsClosing(false)
    }, 300)
  }

  const handleOpen = () => {
    setShowTooltip(false)
    setTooltipDismissed(true)
    setIsOpen(true)
  }

  const handleTooltipDismiss = () => {
    setShowTooltip(false)
    setTooltipDismissed(true)
  }

  const sendVoiceMessage = async (text: string) => {
  if (!text.trim()) return

  // NO inputValue mutation
  setIsTyping(true)

  const userMessage: Message = {
    id: Date.now().toString(),
    text,
    sender: "user",
    timestamp: new Date(),
  }

  setMessages((prev) => [...prev, userMessage])

  const botMessageId = (Date.now() + 1).toString()
  setMessages((prev) => [
    ...prev,
    {
      id: botMessageId,
      text: "",
      sender: "bot",
      timestamp: new Date(),
      sources: [],
    },
  ])

  abortControllerRef.current?.abort()
  abortControllerRef.current = new AbortController()

  const apiUrl =
    process.env.NEXT_PUBLIC_FLASK_API_URL || "http://localhost:8000"

  const response = await fetch(`${apiUrl}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      question: text,
      thread_id: threadId,
      stream: true,
    }),
    signal: abortControllerRef.current.signal,
  })

  const reader = response.body!.getReader()
  const decoder = new TextDecoder()

  let accumulated = ""

  while (true) {
    const { done, value } = await reader.read()
    if (done) break

    const lines = decoder.decode(value).split("\n")

    for (const line of lines) {
      if (!line.startsWith("data: ")) continue
      const data = JSON.parse(line.slice(6))

      if (data.type === "chunk") {
        accumulated += data.data
        setMessages((prev) =>
          prev.map((m) =>
            m.id === botMessageId ? { ...m, text: accumulated } : m
          )
        )
      }

      if (data.type === "done") {
        setIsTyping(false)

        if (!voiceAssistantEnabled && accumulated.trim()) {
          await speakText(accumulated)
        }
      }
    }
  }
}

  const startRecordingBase = async (autoStop: boolean) => {
  if (streamRef.current) {
    streamRef.current.getTracks().forEach((t) => t.stop())
    streamRef.current = null
  }

  const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
  streamRef.current = stream

  const audioContext = new AudioContext()
  const source = audioContext.createMediaStreamSource(stream)
  const analyser = audioContext.createAnalyser()
  analyser.fftSize = 2048
  source.connect(analyser)

  const recorder = new MediaRecorder(stream, {
    mimeType: "audio/webm;codecs=opus",
  })

  mediaRecorderRef.current = recorder
  audioChunksRef.current = []

  recorder.ondataavailable = (e) => {
    if (e.data.size > 0) audioChunksRef.current.push(e.data)
  }

  let silenceStart: number | null = null
  const SILENCE_THRESHOLD = 0.01   // sensitivity
  const SILENCE_DURATION = 800     // ms of silence before stop

  const detectSilence = () => {
    if (!autoStop || recorder.state !== "recording") return

    const buffer = new Uint8Array(analyser.fftSize)
    analyser.getByteTimeDomainData(buffer)

    let sum = 0
    for (let i = 0; i < buffer.length; i++) {
      const val = (buffer[i] - 128) / 128
      sum += val * val
    }

    const volume = Math.sqrt(sum / buffer.length)

    if (volume < SILENCE_THRESHOLD) {
      if (!silenceStart) silenceStart = Date.now()
      if (Date.now() - silenceStart > SILENCE_DURATION) {
        recorder.stop()
        setIsRecording(false)
        return
      }
    } else {
      silenceStart = null
    }

    requestAnimationFrame(detectSilence)
  }

  recorder.onstop = async () => {
    try {
      audioContext.close()

      if (streamRef.current) {
        streamRef.current.getTracks().forEach((t) => t.stop())
        streamRef.current = null
      }

      const audioBlob = new Blob(audioChunksRef.current, {
        type: "audio/webm;codecs=opus",
      })

      if (!audioBlob.size) return

      const apiUrl =
        process.env.NEXT_PUBLIC_FLASK_API_URL || "http://localhost:8000"

      const res = await fetch(`${apiUrl}/transcribe`, {
        method: "POST",
        body: audioBlob,
        headers: { "Content-Type": "audio/webm" },
      })

      const data = await res.json()
      if (!data?.text?.trim()) return

      if (voiceLoopActiveRef.current) {
        await sendVoiceMessage(data.text)
      } else {
        setInputValue(data.text)
        setTimeout(() => handleSend(), 0)
      }
    } catch (err) {
      console.error("Recording failed:", err)
    }
  }

  recorder.start()
  setIsRecording(true)

  if (autoStop) {
    requestAnimationFrame(detectSilence)
  }
}



  

  const stopRecording = async () => {
    const recorder = mediaRecorderRef.current
    if (!recorder) return

    recorder.onstop = async () => {
      console.log("⏹ Recording stopped")

      const audioBlob = new Blob(audioChunksRef.current, { type: "audio/webm;codecs=opus" })
      console.log("📦 Final audio blob size:", audioBlob.size)

      // Stop mic tracks after recording ends
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((t) => t.stop())
        streamRef.current = null
      }

      const apiUrl = process.env.NEXT_PUBLIC_FLASK_API_URL || "http://localhost:8000"

      const res = await fetch(`${apiUrl}/transcribe`, {
        method: "POST",
        body: audioBlob,
        headers: {
          "Content-Type": "audio/webm",
        },
      })

      const data = await res.json()
      if (!data?.text) return

      setInputValue(data.text)
      setTimeout(() => handleSend(), 0)
    }

    recorder.stop()
  }

  const handleMicToggle = async () => {
  if (voiceAssistantEnabled) return

  try {
    if (!isRecording) {
      await startRecordingBase(false) // 👈 manual
    } else {
      await stopRecording()
      setIsRecording(false)
    }
  } catch (e) {
    console.error(e)
  }
}


  const toggleVoiceAssistant = async () => {
  if (voiceAssistantEnabled) {
    setVoiceAssistantEnabled(false);
    return;
  }

  try {
    const apiUrl =
      process.env.NEXT_PUBLIC_FLASK_API_URL || "http://localhost:8000";

    const res = await fetch(`${apiUrl}/livekit-token`);
    const data = await res.json();

    setLivekitToken(data.token);

// small delay so mic publishes before agent attaches
    setTimeout(() => {
      setVoiceAssistantEnabled(true);
    }, 300);

  } catch (e) {
    console.error("Failed to start voice assistant", e);
  }
};








  const toggleSources = (messageId: string) => {
    setExpandedSources((prev) => ({ ...prev, [messageId]: !prev[messageId] }))
  }

  return (
    <>
      {showTooltip && !isOpen && !tooltipDismissed && (
        <div className="fixed bottom-24 right-6 z-50 animate-slide-up-fade">
          <div className="relative bg-card border border-primary/20 rounded-lg shadow-lg p-3 pr-8 max-w-[200px]">
            <p className="text-sm text-foreground font-medium">Click here to chat with our assistant</p>
            <button
              onClick={handleTooltipDismiss}
              className="absolute top-2 right-2 text-muted-foreground hover:text-foreground transition-colors"
              aria-label="Dismiss tooltip"
            >
              <X className="h-3 w-3" />
            </button>
            <div className="absolute -bottom-2 right-8 w-4 h-4 bg-card border-r border-b border-primary/20 transform rotate-45" />
          </div>
        </div>
      )}

      {/* Floating chat button */}
      <Button
        onClick={handleOpen}
        className={
          "fixed bottom-6 right-6 h-14 w-14 rounded-full shadow-lg bg-secondary text-secondary-foreground hover:bg-secondary/90 hover:scale-110 transition-all duration-300 z-50 " +
          (isOpen ? "scale-0" : "scale-100")
        }
        aria-label="Open chat"
      >
        <MessageCircle className="h-6 w-6" />
      </Button>

      {/* Chat panel */}
      {isOpen && (
        <Card
          className={
            "fixed inset-0 w-screen h-screen shadow-2xl z-50 flex flex-col border-none overflow-hidden bg-white " +
            (isClosing ? "animate-scale-out" : "animate-scale-in")
          }
          style={{ borderRadius: "0px" }}
        >
          {/* Header */}
          <div className="flex items-center justify-between p-5 border-b bg-primary text-primary-foreground">
            <div className="flex items-center gap-3">
              <div className="h-10 w-10 rounded-full bg-secondary flex items-center justify-center">
                <Bot className="h-6 w-6 text-secondary-foreground" />
              </div>
              <div>
                <h3 className="font-semibold">LMKR Assistant</h3>
                <p className="text-xs text-primary-foreground/80">Online</p>
              </div>
            </div>
            <Button
              onClick={handleClose}
              variant="ghost"
              size="icon"
              className="h-8 w-8 text-primary-foreground hover:bg-primary-foreground/20 transition-all duration-300 hover:rotate-90"
              aria-label="Close chat"
            >
              <X className="h-5 w-5" />
            </Button>
          </div>

          {/* Messages */}
          <div className="flex-1 overflow-y-auto p-4 space-y-4 bg-white">
            {messages.map((message, index) => {
              const isUser = message.sender === "user"
              const rowClass = "flex gap-3 " + (isUser ? "flex-row-reverse" : "flex-row")

              const bubbleClass =
                "rounded-lg p-3 transition-all duration-200 hover:shadow-md " +
                (isUser
                  ? "bg-primary text-primary-foreground"
                  : "bg-gray-100 text-gray-900 border border-gray-200")

              return (
                <div
                  key={message.id}
                  className={rowClass}
                  style={{
                    animation: `slideUpFadeIn 0.4s cubic-bezier(0.16, 1, 0.3, 1) ${index * 0.05}s backwards`,
                  }}
                >
                  <div
                    className={
                      "h-8 w-8 rounded-full flex items-center justify-center flex-shrink-0 " +
                      (isUser ? "bg-primary text-primary-foreground" : "bg-secondary text-secondary-foreground")
                    }
                  >
                    {isUser ? <User className="h-4 w-4" /> : <Bot className="h-4 w-4" />}
                  </div>

                  <div className="flex flex-col gap-2 max-w-[70%]">
                    <div className={bubbleClass}>
                      {message.sender === "bot" ? (
  <ReactMarkdown
    remarkPlugins={[remarkGfm]}
    components={{
      p: ({ children }) => (
        <p className="text-sm leading-relaxed mb-2">{children}</p>
      ),
      li: ({ children }) => (
        <li className="ml-5 list-disc text-sm mb-1">{children}</li>
      ),
      ol: ({ children }) => (
        <ol className="ml-5 list-decimal mb-2">{children}</ol>
      ),
      ul: ({ children }) => (
        <ul className="ml-5 list-disc mb-2">{children}</ul>
      ),
      strong: ({ children }) => (
        <strong className="font-semibold">{children}</strong>
      ),
      a: ({ href, children }) => (
        <a
          href={href}
          target="_blank"
          rel="noreferrer"
          className="text-primary underline break-all"
        >
          {children}
        </a>
      ),
    }}
  >
    {message.text}
  </ReactMarkdown>
) : (
  <p className="text-sm leading-relaxed">{message.text}</p>
)}


                      {/* 🔊 Per-bot-message speaker button */}
                      {!isUser && message.text?.trim() && (
                        <button
                          onClick={() => {
                            if (!audioRef.current || audioRef.current.paused) {
                              speakText(message.text)
                            } else {
                              stopTtsPlayback()
                            }
                          }}

                          className="mt-2 inline-flex items-center gap-1 text-xs text-primary hover:text-primary/80 transition-colors"
                          aria-label="Speak this message"
                        >
                          <Volume2 className="h-3 w-3" />
                          Speak
                        </button>
                      )}

                      <p className={"text-xs mt-1 " + (isUser ? "text-primary-foreground/70" : "text-gray-500")}>
                        {message.timestamp.toLocaleTimeString([], {
                          hour: "2-digit",
                          minute: "2-digit",
                        })}
                      </p>
                    </div>

                    {message.sources && message.sources.length > 0 && (
                      <div className="flex flex-col gap-1">
                        <button
                          onClick={() => toggleSources(message.id)}
                          className="flex items-center gap-1 text-xs text-primary hover:text-primary/80 transition-colors duration-200"
                        >
                          {expandedSources[message.id] ? (
                            <ChevronUp className="h-3 w-3" />
                          ) : (
                            <ChevronDown className="h-3 w-3" />
                          )}
                          View sources ({message.sources.length})
                        </button>

                        {expandedSources[message.id] && (
                          <div
                            className="bg-gray-50 border border-gray-200 rounded-lg p-2 space-y-1"
                            style={{
                              animation: "slideDownFadeIn 0.3s cubic-bezier(0.16, 1, 0.3, 1) backwards",
                            }}
                          >
                            {message.sources.map((source, idx) => (
                              <a
                                key={idx}
                                href={source.url}
                                className="block text-xs text-primary hover:text-secondary transition-colors duration-200 hover:underline"
                                target="_blank"
                                rel="noreferrer"
                              >
                                {idx + 1}. {source.title}
                              </a>
                            ))}
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                </div>
              )
            })}

            {isTyping && (
              <div className="flex gap-3" style={{ animation: "fadeIn 0.3s ease-out" }}>
                <div className="h-8 w-8 rounded-full flex items-center justify-center flex-shrink-0 bg-secondary text-secondary-foreground">
                  <Bot className="h-4 w-4" />
                </div>
                <div className="bg-gray-100 border border-gray-200 rounded-lg p-3 flex items-center gap-1">
                  <div className="w-2 h-2 bg-gray-400 rounded-full" style={{ animation: "bounce 1s infinite 0s" }} />
                  <div className="w-2 h-2 bg-gray-400 rounded-full" style={{ animation: "bounce 1s infinite 0.2s" }} />
                  <div className="w-2 h-2 bg-gray-400 rounded-full" style={{ animation: "bounce 1s infinite 0.4s" }} />
                </div>
              </div>
            )}

            <div ref={messagesEndRef} />
          </div>

          {/* Input */}
          <div className="p-4 border-t border-gray-200 bg-white">
            <div className="flex gap-2">
              <Input
                value={inputValue}
                onChange={(e) => setInputValue(e.target.value)}
                onKeyPress={handleKeyPress}
                placeholder="Type your message..."
                className="flex-1 bg-white border-gray-300 text-gray-900 placeholder:text-gray-500"
              />

                {/* 🧠 Voice Assistant Toggle */}
                <Button
                  onClick={toggleVoiceAssistant}
                  size="icon"
                  variant="outline"
                  className={
                    voiceAssistantEnabled
                      ? "bg-green-500 text-white animate-pulse"
                      : ""
                 }
                 aria-label="Voice assistant"
>
  🧠
</Button>


              {/* 🎙 Mic */}
              <Button
                onClick={handleMicToggle}
                size="icon"
                variant="outline"
                className={
                  isRecording
                    ? "bg-secondary text-secondary-foreground border-secondary animate-pulse"
                    : "bg-white text-gray-700 border-gray-300 hover:bg-gray-50"
                }
                aria-label="Voice input"
              >
                <Mic className="h-4 w-4" />
              </Button>

              {/* 📤 Send */}
              <Button
                onClick={handleSend}
                size="icon"
                className="bg-secondary text-secondary-foreground hover:bg-secondary/90 hover:scale-105 transition-all duration-300"
                aria-label="Send message"
              >
                <Send className="h-4 w-4" />
              </Button>
            </div>
          </div>
        </Card>
      )}

      <VoiceAssistantLiveKit
      enabled={voiceAssistantEnabled}
      token={livekitToken}
/>






      <style jsx>{`
  /* ============================
     CHAT OPEN / CLOSE ANIMATIONS
     ============================ */

  @keyframes scaleIn {
    from {
      opacity: 0;
      transform: scale(0.96);
    }
    to {
      opacity: 1;
      transform: scale(1);
    }
  }

  @keyframes scaleOut {
    from {
      opacity: 1;
      transform: scale(1);
    }
    to {
      opacity: 0;
      transform: scale(0.96);
    }
  }

  .animate-scale-in {
    animation: scaleIn 0.3s cubic-bezier(0.16, 1, 0.3, 1) forwards;
  }

  .animate-scale-out {
    animation: scaleOut 0.25s ease-in forwards;
  }

  /* ============================
     MESSAGE ENTRY ANIMATIONS
     ============================ */

  @keyframes slideUpFadeIn {
    from {
      opacity: 0;
      transform: translateY(16px);
    }
    to {
      opacity: 1;
      transform: translateY(0);
    }
  }

  @keyframes slideDownFadeIn {
    from {
      opacity: 0;
      transform: translateY(-10px);
    }
    to {
      opacity: 1;
      transform: translateY(0);
    }
  }

  @keyframes fadeIn {
    from {
      opacity: 0;
    }
    to {
      opacity: 1;
    }
  }

  /* ============================
     TYPING INDICATOR
     ============================ */

  @keyframes bounce {
    0%,
    100% {
      transform: translateY(0);
    }
    50% {
      transform: translateY(-6px);
    }
  }

  /* ============================
     TOOLTIP ANIMATION
     ============================ */

  @keyframes slideUpFade {
    from {
      opacity: 0;
      transform: translateY(20px);
    }
    to {
      opacity: 1;
      transform: translateY(0);
    }
  }

  .animate-slide-up-fade {
    animation: slideUpFade 0.35s cubic-bezier(0.16, 1, 0.3, 1) forwards;
  }

  /* ============================
     OPTIONAL SMOOTH SCROLL
     ============================ */

  .chat-scroll {
    scroll-behavior: smooth;
  }

  /* ============================
     MOBILE FULLSCREEN FIX
     ============================ */

  @media (max-width: 640px) {
    .chat-fullscreen {
      border-radius: 0 !important;
    }
  }
`}</style>

    </>
  )
}
