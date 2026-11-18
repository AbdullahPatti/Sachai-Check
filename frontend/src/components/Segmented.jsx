// Pill switch with a sliding indicator. options: [{ v, label }]
export default function Segmented({ options, value, onChange }) {
  const idx = Math.max(0, options.findIndex((o) => o.v === value));
  return (
    <div className="seg">
      <span className="ind" aria-hidden="true" style={{ transform: `translateX(${idx * 100}%)` }} />
      {options.map((o) => (
        <button key={o.v} type="button" aria-pressed={o.v === value} onClick={() => onChange(o.v)}>
          {o.label}
        </button>
      ))}
    </div>
  );
}
