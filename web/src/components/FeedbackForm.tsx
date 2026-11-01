import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useState, type FormEvent } from "react";
import { api, queryKeys, type IncidentDetail } from "../api/client";
import { Button } from "./Button";
import { Select } from "./Fields";
import styles from "./decision.module.css";
import primitives from "./primitives.module.css";

const UNKNOWN = "";

export function FeedbackForm({ incident }: { incident: IncidentDetail }) {
  const client = useQueryClient();
  const [correct, setCorrect] = useState<boolean | null>(null);
  const [actual, setActual] = useState(UNKNOWN);
  const [note, setNote] = useState("");
  const top = incident.top_service;
  const others = incident.diagnosis.ranking
    .map((item) => item.service)
    .filter((service) => service !== top);

  const save = useMutation({
    mutationFn: () =>
      api.feedback(incident.id, {
        correct: correct === true,
        actual_service: correct || actual === UNKNOWN ? null : actual,
        note,
      }),
    onSuccess: () =>
      Promise.all([
        client.invalidateQueries({ queryKey: queryKeys.incident(incident.id) }),
        client.invalidateQueries({ queryKey: ["incidents"] }),
        client.invalidateQueries({ queryKey: queryKeys.history }),
      ]),
  });

  function submit(event: FormEvent) {
    event.preventDefault();
    save.mutate();
  }

  return (
    <form className={styles.form} onSubmit={submit} aria-label="Feedback">
      {incident.status === "closed" && (
        <p className={styles.saved}>A label is saved. Submitting again replaces it.</p>
      )}
      <fieldset className={styles.choices}>
        <legend>
          Was <span className="mono">{top}</span> the origin?
        </legend>
        <label>
          <input
            type="radio"
            name="correct"
            checked={correct === true}
            onChange={() => setCorrect(true)}
          />
          Yes
        </label>
        <label>
          <input
            type="radio"
            name="correct"
            checked={correct === false}
            onChange={() => setCorrect(false)}
          />
          No
        </label>
      </fieldset>
      {correct === false && (
        <Select
          label="Confirmed origin"
          value={actual}
          onChange={(event) => setActual(event.target.value)}
          options={[
            { value: UNKNOWN, label: "Not known" },
            ...others.map((service) => ({ value: service, label: service })),
          ]}
        />
      )}
      <label className={primitives.field}>
        <span className={primitives.fieldLabel}>Note</span>
        <textarea
          className={primitives.control}
          rows={2}
          maxLength={2000}
          value={note}
          onChange={(event) => setNote(event.target.value)}
        />
      </label>
      <div>
        <Button variant="primary" type="submit" disabled={correct === null || save.isPending}>
          Save label
        </Button>
      </div>
      {save.isError && <p role="alert">{save.error.message}</p>}
      {save.isSuccess && (
        <p role="status" className={styles.saved}>
          Label saved.
        </p>
      )}
    </form>
  );
}
