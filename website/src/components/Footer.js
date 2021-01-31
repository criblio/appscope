import React from "react";
import { Link } from "gatsby";
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
          <Col xs={12} md={6} className="text-left">
            <Link to="https://cribl.io">
              <img src={logo} alt="Cribl" width={125} />
            </Link>
          </Col>
          <Col xs={12} md={6} className="text-right footer-right">
            <p>Cribl, &copy; 2021</p>
            <Link to="https://www.facebook.com/Cribl-258234158133458/">
              <FontAwesomeIcon icon={["fab", "facebook"]} />
            </Link>
            <Link to="https://twitter.com/cribl_io">
              <FontAwesomeIcon icon={["fab", "twitter"]} />
            </Link>
            <Link to="https://www.linkedin.com/company/18777798">
              <FontAwesomeIcon icon={["fab", "linkedin"]} />
            </Link>
          </Col>
        </Row>
      </Container>
    </Container>
  );
}
