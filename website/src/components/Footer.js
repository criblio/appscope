import React from "react";
import { Container, Row, Col } from "react-bootstrap";
import logo from "../images/logo-cribl-new.svg";
import "../scss/_footer.scss";

export default function Footer() {
  return (
    <Container fluid className="footer-container">
      <Container>
        <Row>
          <Col xs={12} md={2}>
            <img src={logo} alt="Cribl" width={125} />
          </Col>
          <Col xs={12} md={8} className="align-items-center">
            <p>Cribl, &copy; 2021</p>
          </Col>
        </Row>
      </Container>
    </Container>
  );
}
