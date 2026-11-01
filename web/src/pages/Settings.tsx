import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState, type FormEvent } from "react";
import { api, queryKeys, type IntegrationList, type SettingsView } from "../api/client";
import { Button } from "../components/Button";
import { Card, CardBody, CardHeader } from "../components/Card";
import { TextInput, Toggle } from "../components/Fields";
import { QueryView } from "../components/QueryView";
import { Tag } from "../components/Tag";
import { formatUnit } from "../format";
import styles from "./pages.module.css";

const SECURITY = [
  "No sign-in in this release. Keep the service on this machine and never expose it publicly.",
  "It binds to 127.0.0.1 by default and warns if bound to every interface.",
  "It only reads the cases given to it, and it makes no outbound requests.",
  "Nothing here changes infrastructure or calls production systems.",
];

const optional = (text: string) => (text.trim() === "" ? null : Number(text));

function Sources({ list }: { list: IntegrationList }) {
  return (
    <>
      <p className={styles.meta}>{list.read_only}</p>
      <div className={styles.pair}>
        {list.items.map((item) => (
          <Card key={item.name}>
            <CardHeader
              title={item.name}
              actions={<Tag value="Configuration only" tone="warn" />}
            />
            <CardBody>
              <p className={styles.meta}>
                {item.kind}. {item.configured ? "Configured" : "Not configured"}, not connected.
              </p>
            </CardBody>
          </Card>
        ))}
      </div>
    </>
  );
}

function DecisionForm({ settings }: { settings: SettingsView }) {
  const client = useQueryClient();
  const [threshold, setThreshold] = useState(settings.abstain_threshold?.toString() ?? "");
  const [windowLength, setWindowLength] = useState(settings.window_seconds?.toString() ?? "");
  const [provided, setProvided] = useState(settings.use_provided_topology ?? true);
  const save = useMutation({
    mutationFn: () =>
      api.saveSettings({
        abstain_threshold: optional(threshold),
        window_seconds: optional(windowLength),
        use_provided_topology: provided,
      }),
    onSuccess: () => client.invalidateQueries({ queryKey: queryKeys.settings }),
  });
  const model = settings.model_threshold;

  function submit(event: FormEvent) {
    event.preventDefault();
    save.mutate();
  }

  return (
    <form className={styles.stack} onSubmit={submit} aria-label="Decision settings">
      <TextInput
        label={`Abstain threshold, 0 to 1. Blank uses the model's${
          model == null ? "" : ` (${formatUnit(model)})`
        }`}
        type="number"
        min={0}
        max={1}
        step="any"
        value={threshold}
        onChange={(event) => setThreshold(event.target.value)}
      />
      <TextInput
        label="Window length, 60 to 3600 seconds. Blank uses each case's whole window"
        type="number"
        min={60}
        max={3600}
        step={1}
        value={windowLength}
        onChange={(event) => setWindowLength(event.target.value)}
      />
      <Toggle label="Use provided call graphs" checked={provided} onChange={setProvided} />
      <p className={styles.meta}>
        When off, the model ignores a provided call graph and may infer one. Changes apply to new
        and re-run diagnoses, never to stored ones.
      </p>
      <div>
        <Button variant="primary" type="submit" disabled={save.isPending}>
          Save settings
        </Button>
      </div>
      {save.isError && <p role="alert">{save.error.message}</p>}
      {save.isSuccess && <p role="status">Settings saved.</p>}
    </form>
  );
}

export function Settings() {
  const integrations = useQuery({
    queryKey: queryKeys.integrations,
    queryFn: ({ signal }) => api.integrations(signal),
  });
  const settings = useQuery({
    queryKey: queryKeys.settings,
    queryFn: ({ signal }) => api.settings(signal),
  });
  return (
    <div className={styles.page}>
      <h1 className="visually-hidden">Integrations and settings</h1>
      <Card>
        <CardHeader title="Data sources" />
        <CardBody>
          <QueryView query={integrations} label="integrations">
            {(list) => <Sources list={list} />}
          </QueryView>
        </CardBody>
      </Card>
      <div className={styles.pair}>
        <Card>
          <CardHeader title="Decision settings" />
          <CardBody>
            <QueryView query={settings} label="settings">
              {(data) => <DecisionForm settings={data} />}
            </QueryView>
          </CardBody>
        </Card>
        <Card>
          <CardHeader title="Security and access" />
          <CardBody>
            <ul className={styles.notes}>
              {SECURITY.map((line) => (
                <li key={line}>{line}</li>
              ))}
            </ul>
          </CardBody>
        </Card>
      </div>
    </div>
  );
}
