import React from "react";
import { Container, Row, Col } from "react-bootstrap";

import "../../utils/font-awesome";
export default function Layout({ children }) {
  return (
    <Container className="community-layout">
      <h2>Community</h2>
      <p>Lorem Ipsum dolor sit amet</p>

      <Row>
        <Container className="community-cards">{children}</Container>
      </Row>
    </Container>
  );
}
