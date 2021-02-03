import React from "react";
import { Container, Row, Col } from "react-bootstrap";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import logo from "../images/logo-cribl-new.svg";
import "../scss/_footer.scss";
import "../utils/font-awesome";

export default function Footer() {
  return (
    <Container fluid className="footer-container ">
      <Container>
        <Row>
          <Col xs={12} md={6} className="text-left footer-left">
            <a href="https://cribl.io">
              <img src={logo} alt="Cribl" width={125} />
            </a>
          </Col>
          <Col xs={12} md={6} className="text-right footer-right">
            <p>Cribl, &copy; 2021</p>
            <a href="https://www.facebook.com/Cribl-258234158133458/">
              <FontAwesomeIcon icon={["fab", "facebook"]} />
            </a>
            <a href="https://twitter.com/cribl_io">
              <FontAwesomeIcon icon={["fab", "twitter"]} />
            </a>
            <a href="https://www.linkedin.com/company/18777798">
              <FontAwesomeIcon icon={["fab", "linkedin"]} />
            </a>
          </Col>
        </Row>
      </Container>
    </Container>
  );
}
