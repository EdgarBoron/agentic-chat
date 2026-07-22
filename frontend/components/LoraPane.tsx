import type { LoraConfig } from "@/lib/lora-config";

export function LoraPane({
  loras,
  onLorasChange,
}: {
  loras: LoraConfig[];
  onLorasChange: (loras: LoraConfig[]) => void;
}) {
  function toggleLora(id: string, enabled: boolean) {
    onLorasChange(loras.map((l) => (l.id === id ? { ...l, enabled } : l)));
  }

  function setLoraWeight(id: string, weight: number) {
    onLorasChange(loras.map((l) => (l.id === id ? { ...l, weight } : l)));
  }

  return (
    <aside className="lora-pane">
      <div className="lora-pane-header">LoRAs</div>
      <div className="lora-list">
        {loras.map((lora) => (
          <label key={lora.id} className="lora-item">
            <input
              type="checkbox"
              checked={lora.enabled}
              onChange={(e) => toggleLora(lora.id, e.target.checked)}
            />
            <span className="lora-label">{lora.label}</span>
            <input
              type="number"
              min={0}
              max={2}
              step={0.05}
              value={lora.weight}
              disabled={!lora.enabled}
              onChange={(e) => setLoraWeight(lora.id, Number(e.target.value))}
              className="lora-weight"
            />
          </label>
        ))}
      </div>
    </aside>
  );
}
