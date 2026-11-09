import { useEffect, useRef, useState } from 'react';

export function OpacityControl({
  label,
  value,
  disabled,
  preview,
  commit,
}: {
  label: string;
  value: number;
  disabled: boolean;
  preview: (value: number) => void;
  commit: (value: number) => Promise<void>;
}) {
  const [draft, setDraft] = useState(value);
  const saved = useRef(value);
  useEffect(() => {
    setDraft(value);
    saved.current = value;
  }, [value]);
  const apply = () => {
    if (saved.current !== draft) {
      saved.current = draft;
      void commit(draft);
    }
  };
  return (
    <label className="opacity-control">
      <span>
        {label}
        <output>{Math.round(draft * 100)}%</output>
      </span>
      <input
        aria-label={label}
        type="range"
        min="0"
        max="1"
        step="0.05"
        value={draft}
        disabled={disabled}
        onChange={(event) => {
          const next = Number(event.target.value);
          setDraft(next);
          preview(next);
        }}
        onPointerUp={apply}
        onKeyUp={apply}
        onBlur={apply}
      />
    </label>
  );
}
