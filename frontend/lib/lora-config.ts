export type LoraConfig = {
  id: string;
  label: string;
  file: string;
  enabled: boolean;
  weight: number;
};

// Mirrors the three LoRAs enabled by default in the reference ComfyUI
// workflow's Power Lora Loader node (zImage-prompt-gen.json), same filenames
// and strengths.
export const DEFAULT_LORAS: LoraConfig[] = [
  { id: "midj-z-1", label: "Midj Z-1", file: "midj-z-1.safetensors", enabled: true, weight: 0.35 },
  {
    id: "zidius-melancholy",
    label: "Zidius Art (melancholy)",
    file: "zImageT_zidiusArt_melancholy.safetensors",
    enabled: true,
    weight: 0.25,
  },
  {
    id: "purple-grainy",
    label: "Purple Grainy",
    file: "Purple_grainy_zit.safetensors",
    enabled: true,
    weight: 0.4,
  },
];
