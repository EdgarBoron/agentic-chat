export type SlashCommand = {
  name: string;
  description: string;
};

export const SLASH_COMMANDS: SlashCommand[] = [
  { name: "help", description: "List available commands" },
  { name: "clear", description: "Clear the current conversation and start a new session" },
];

export function matchCommands(input: string): SlashCommand[] {
  if (!input.startsWith("/")) return [];
  const query = input.slice(1).toLowerCase();
  return SLASH_COMMANDS.filter((c) => c.name.startsWith(query));
}

export function findExactCommand(input: string): SlashCommand | undefined {
  const name = input.trim().slice(1).toLowerCase();
  return SLASH_COMMANDS.find((c) => c.name === name);
}

export function helpText(): string {
  return SLASH_COMMANDS.map((c) => `/${c.name} — ${c.description}`).join("\n");
}
