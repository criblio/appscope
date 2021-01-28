import React, { useEffect } from "react";
import { Container, Row, Col } from "react-bootstrap";
import DocsNav from "../DocsNav";
import Header from "../Header";
import MobileHeader from "../MobileHeader";
import "../../scss/_documentation.scss";
export default function Layout({ children }) {
  return (
    <>
      <div className="display-xs">
        <MobileHeader />
      </div>

      <div className="display-md">
        <Header />
      </div>
      <Container
        style={{ paddingTop: 50 }}
        data-spy="affix"
        data-offset-top="197"
      >
        <Row className=" align-items-start">
          <Col md={3} xs={12} className="position-fixed">
            <DocsNav />
          </Col>
          <Col
            md={{ span: 9, offset: 3 }}
            xs={12}
            className="documentation"
            id="docContainer"
          >
            {children}
          </Col>
          {/* <Col md={3} className="TOC">
            <ul>
              <li>Introduction</li>
              <li>Requirements</li>
              <li>Install and Build</li>
              <li>Developer Notes</li>
              <li>Environment Variables</li>
              <li>Features</li>
            </ul>
          </Col> */}
        </Row>
      </Container>
    </>
  );
}
