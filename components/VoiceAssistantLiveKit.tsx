"use client";

import type { LocalTrackPublication } from "livekit-client";
import { useEffect, useRef } from "react";
import {
  Room,
  RoomEvent,
  Track,
  createLocalAudioTrack,
  LocalAudioTrack,
} from "livekit-client";

type Props = {
  enabled: boolean;
  token: string | null;
};

export function VoiceAssistantLiveKit({ enabled, token }: Props) {
  const roomRef = useRef<Room | null>(null);

  useEffect(() => {
    if (!enabled || !token) return;

    const room = new Room({
      adaptiveStream: true,
      dynacast: true,
      publishDefaults: {
        audioRedundantEncoding: false, // ✅ THIS disables audio/red
  } as any,
});



    roomRef.current = room;

    /* ------------------ ROOM EVENTS ------------------ */

    room.on(RoomEvent.Connected, () => {
      console.log("✅ Browser connected to LiveKit room");
    });

    room.on(RoomEvent.Disconnected, () => {
      console.log("❌ Browser disconnected from LiveKit");
    });

    room.on(RoomEvent.TrackSubscribed, (track, _pub, participant) => {
      if (
        track.kind === Track.Kind.Audio &&
        participant.identity !== room.localParticipant?.identity
      ) {
        console.log("🔊 Playing agent audio");

        const el = track.attach() as HTMLAudioElement;
        el.autoplay = true;
        el.setAttribute("playsinline", "true");
        document.body.appendChild(el);
      }
    });

    /* ------------------ MAIN FLOW ------------------ */

    (async () => {
      try {
        console.log("🔌 Connecting to LiveKit...");
        await room.connect(
          process.env.NEXT_PUBLIC_LIVEKIT_URL!,
          token
        );

        console.log("🎙 Creating microphone track...");

        const micTrack: LocalAudioTrack =
          await createLocalAudioTrack({
            echoCancellation: true,
            noiseSuppression: true,
            autoGainControl: true,
          });

        micTrack.mediaStreamTrack.enabled = true;

        await room.localParticipant.publishTrack(micTrack);
        console.log("🎙 Microphone published to LiveKit");
      } catch (e) {
        console.error("❌ LiveKit error", e);
      }
    })();

    /* ------------------ CLEANUP ------------------ */

    return () => {
      console.log("🧹 Cleaning up LiveKit");

      const participant = room.localParticipant as any;

      if (participant?.audioTracks) {
        for (const pub of participant.audioTracks.values() as Iterable<LocalTrackPublication>) {
          const track = pub.track;
          if (track) {
            track.stop();
            participant.unpublishTrack(track);
          }
        }
      }

      room.disconnect();
      roomRef.current = null;
    };
  }, [enabled, token]);

  return null;
}
