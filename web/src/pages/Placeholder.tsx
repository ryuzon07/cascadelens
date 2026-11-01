import { Card, CardBody, CardHeader } from "../components/Card";
import { EmptyState } from "../components/States";

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
