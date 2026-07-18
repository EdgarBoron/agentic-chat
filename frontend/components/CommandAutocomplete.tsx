import type { SlashCommand } from "@/lib/commands";

export function CommandAutocomplete({
  commands,
  highlightedIndex,
  onSelect,
}: {
  commands: SlashCommand[];
  highlightedIndex: number;
  onSelect: (command: SlashCommand) => void;
}) {
  if (commands.length === 0) return null;

  return (
    <div className="command-autocomplete">
      {commands.map((cmd, i) => (
        <button
          type="button"
          key={cmd.name}
          className={`command-item ${i === highlightedIndex ? "highlighted" : ""}`}
          onMouseDown={(e) => {
            // onMouseDown (not onClick) so it fires before the input's blur.
            e.preventDefault();
            onSelect(cmd);
          }}
        >
          <span className="command-name">/{cmd.name}</span>
          <span className="command-description">{cmd.description}</span>
        </button>
      ))}
    </div>
  );
}
