import React from "react";
import { Container, Row, Col } from "react-bootstrap";

import "../../utils/font-awesome";
export default function Layout({ children }) {
  return (
    <Container>
      <Row>
        <h2>Community</h2>
        <p>Lorem Ipsum dolor sit amet</p>
        <Col xs={12} md={6}>
          <Container>{children}</Container>
        </Col>
      </Row>
    </Container>
  );
}
