import { forwardRef, type InputHTMLAttributes, type SelectHTMLAttributes } from "react";
import styles from "./primitives.module.css";

interface TextInputProps extends InputHTMLAttributes<HTMLInputElement> {
  label: string;
  hideLabel?: boolean;
}

export const TextInput = forwardRef<HTMLInputElement, TextInputProps>(function TextInput(
  { label, hideLabel, ...props },
  ref,
) {
  return (
    <label className={styles.field}>
      <span className={hideLabel ? "visually-hidden" : styles.fieldLabel}>{label}</span>
      <input ref={ref} className={styles.control} {...props} />
    </label>
  );
});

interface SelectProps extends SelectHTMLAttributes<HTMLSelectElement> {
  label: string;
  options: { value: string; label: string }[];
}

export function Select({ label, options, ...props }: SelectProps) {
  return (
    <label className={styles.field}>
      <span className={styles.fieldLabel}>{label}</span>
      <select className={styles.control} {...props}>
        {options.map((option) => (
          <option key={option.value} value={option.value}>
            {option.label}
          </option>
        ))}
      </select>
    </label>
  );
}

interface SliderProps {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (value: number) => void;
  format?: (value: number) => string;
  disabled?: boolean;
}

export function Slider({ label, value, onChange, format, ...range }: SliderProps) {
  const shown = format ? format(value) : String(value);
  return (
    <label className={styles.field}>
      <span className={styles.sliderRow}>
        <span className={styles.fieldLabel}>{label}</span>
        <span className={styles.sliderValue}>{shown}</span>
      </span>
      <input
        type="range"
        className={styles.slider}
        value={value}
        aria-valuetext={shown}
        onChange={(event) => onChange(Number(event.target.value))}
        {...range}
      />
    </label>
  );
}

interface ToggleProps {
  label: string;
  checked: boolean;
  onChange: (checked: boolean) => void;
  disabled?: boolean;
}

/** A switch. The text beside it states its position, so colour never has to. */
export function Toggle({ label, checked, onChange, disabled }: ToggleProps) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={checked}
      disabled={disabled}
      className={styles.toggle}
      onClick={() => onChange(!checked)}
    >
      <span className={styles.track} aria-hidden="true" />
      <span>{label}</span>
      <span className={styles.fieldLabel}>{checked ? "On" : "Off"}</span>
    </button>
  );
}
