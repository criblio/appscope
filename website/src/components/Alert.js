import React from "react";
import "../scss/_docsNav.scss";
import { Container, Row, Col } from "react-bootstrap";
import "../scss/_alert.scss";

export default function Alert() {
  return (
    <Container fluid className="alert">
      <Row className="align-items-center">
        <Col>
          <p>Alert goes here. I'm sure I don't need to explain this one. </p>
        </Col>
      </Row>
    </Container>
  );
}
