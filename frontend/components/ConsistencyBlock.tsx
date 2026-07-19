"use client";

import { useId, useState } from "react";

type ConsistencyOption = { id: string; label: string };
type ConsistencyIssue = { id: string; summary: string; options: ConsistencyOption[] };
type ConsistencyPayload = { issues: ConsistencyIssue[] };

const IGNORE_OPTION_ID = "__ignore__";
const IGNORE_OPTION_LABEL = "Ignore — leave this as-is";

function parsePayload(content: string): ConsistencyPayload | null {
  let parsed: unknown;
  try {
    parsed = JSON.parse(content);
  } catch {
    return null;
  }
  const issues = (parsed as { issues?: unknown })?.issues;
  if (!Array.isArray(issues) || issues.length === 0) return null;

  const cleaned: ConsistencyIssue[] = [];
  for (const issue of issues) {
    if (
      typeof issue !== "object" ||
      issue === null ||
      typeof (issue as ConsistencyIssue).id !== "string" ||
      typeof (issue as ConsistencyIssue).summary !== "string" ||
      !Array.isArray((issue as ConsistencyIssue).options)
    ) {
      return null;
    }
    const options = (issue as ConsistencyIssue).options.filter(
      (o): o is ConsistencyOption =>
        typeof o === "object" && o !== null && typeof o.id === "string" && typeof o.label === "string"
    );
    if (options.length === 0) return null;
    cleaned.push({ id: (issue as ConsistencyIssue).id, summary: (issue as ConsistencyIssue).summary, options });
  }
  return { issues: cleaned };
}

export function ConsistencyBlock({
  content,
  onSubmit,
}: {
  content: string;
  onSubmit: (text: string) => void;
}) {
  // A per-instance id namespaces radio group `name`s (and selection keys) so
  // that two consistency blocks on the page never collide — the model tends
  // to reuse plain issue ids like "1"/"2" in every check, and duplicate
  // `name` attributes across separate <fieldset> groups make clicking one
  // radio visually toggle a same-named radio in the other group.
  const blockId = useId();
  const [selections, setSelections] = useState<Record<number, string>>({});
  const [submitted, setSubmitted] = useState(false);

  const payload = parsePayload(content);
  if (!payload) {
    // Still streaming in (incomplete JSON) or the model didn't follow the
    // format — fall back to showing the raw text rather than crashing.
    return (
      <div className="consistency-block">
        <pre>{content.trim()}</pre>
      </div>
    );
  }

  const allAnswered = payload.issues.every((_, idx) => selections[idx]);

  function handleSubmit() {
    const lines = payload!.issues.map((issue, idx) => {
      const chosenId = selections[idx];
      if (chosenId === IGNORE_OPTION_ID) {
        return `- ${issue.summary} → leave as-is (ignored)`;
      }
      const chosen = issue.options.find((o) => o.id === chosenId);
      return `- ${issue.summary} → ${chosen?.label ?? ""}`;
    });
    onSubmit(`Apply these choices for the inconsistencies you found:\n${lines.join("\n")}`);
    setSubmitted(true);
  }

  return (
    <div className="consistency-block">
      <div className="consistency-block-header">Inconsistencies found</div>
      {payload.issues.map((issue, idx) => {
        const groupName = `${blockId}-issue-${idx}`;
        return (
          <fieldset className="consistency-issue" key={idx} disabled={submitted}>
            <legend>{issue.summary}</legend>
            {issue.options.map((option) => (
              <label className="consistency-option" key={option.id}>
                <input
                  type="radio"
                  name={groupName}
                  value={option.id}
                  checked={selections[idx] === option.id}
                  onChange={() => setSelections((prev) => ({ ...prev, [idx]: option.id }))}
                />
                {option.label}
              </label>
            ))}
            <label className="consistency-option consistency-option-ignore">
              <input
                type="radio"
                name={groupName}
                value={IGNORE_OPTION_ID}
                checked={selections[idx] === IGNORE_OPTION_ID}
                onChange={() => setSelections((prev) => ({ ...prev, [idx]: IGNORE_OPTION_ID }))}
              />
              {IGNORE_OPTION_LABEL}
            </label>
          </fieldset>
        );
      })}
      <div className="consistency-block-footer">
        {submitted ? (
          <span className="consistency-submitted">Choices submitted</span>
        ) : (
          <button type="button" disabled={!allAnswered} onClick={handleSubmit}>
            Submit
          </button>
        )}
      </div>
    </div>
  );
}
