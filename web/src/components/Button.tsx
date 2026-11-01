import type { ButtonHTMLAttributes, ReactNode } from "react";
import styles from "./primitives.module.css";

interface BaseProps extends Omit<ButtonHTMLAttributes<HTMLButtonElement>, "aria-label"> {
  children: ReactNode;
}

type ButtonProps =
  | (BaseProps & { variant?: "primary" | "secondary" | "ghost"; "aria-label"?: string })
  | (BaseProps & { variant: "icon"; "aria-label": string });

export function Button({ variant = "secondary", className, type, ...props }: ButtonProps) {
  const classes = [styles.button, styles[variant], className].filter(Boolean).join(" ");
  return <button type={type ?? "button"} className={classes} {...props} />;
}
