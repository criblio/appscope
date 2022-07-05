import React from "react";
import { Container, Row, Col } from "react-bootstrap";
import DocsNav from "../DocsNav";
import Header from "../Header";
import MobileHeader from "../MobileHeader";
import { Helmet } from "react-helmet";

import "../../scss/_documentation.scss";
import Footer from "../Footer";
export default function Layout({ children }) {
  return (
    <>
      <Helmet>
        <meta name="og:image" content={'https://cribl.io/wp-content/uploads/2022/01/thumb.appScope.fullColorWhiteAlt.png'} />
      </Helmet>
      <div className="display-xs">
        <MobileHeader />
      </div>

      <div className="display-md">
        <Header />
      </div>
      <Container style={{ paddingTop: 50 }} id="docsContainer">
        <Row className=" align-items-start" style={{ marginTop: 110 }}>
          <Col md={3} xs={12} className="" style={{ zIndex: 1000 }}>
            <DocsNav />
          </Col>
          <Col
            md={{ span: 9 }}
            xs={12}
            className="documentation"
            id="docContainer"
          >
            {children}
          </Col>
        </Row>
      </Container>
      <Footer />
    </>
  );
}
