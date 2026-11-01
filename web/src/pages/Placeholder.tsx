import { Card, CardBody, CardHeader } from "../components/Card";
import { EmptyState } from "../components/States";

export function Placeholder({ title, phase }: { title: string; phase: number }) {
  return (
    <Card>
      <CardHeader title={title} />
      <CardBody>
        <EmptyState title="Not built yet">This screen arrives in UI phase {phase}.</EmptyState>
      </CardBody>
    </Card>
  );
}

export function NotFound() {
  return (
    <Card>
      <CardHeader title="Not found" />
      <CardBody>
        <EmptyState title="This page does not exist" />
      </CardBody>
    </Card>
  );
}
