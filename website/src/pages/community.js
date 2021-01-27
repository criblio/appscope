import * as React from "react";
import Header from "../components/Header";
import Alert from "../components/Alert";
import MobileHeader from "../components/MobileHeader";
import Layout from "../components/layouts/community";
import MarkDownBlock from "../components/MarkDownBlock";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import { Container, Row, Col } from "react-bootstrap";
import "../utils/font-awesome";
const CommunityMain = () => {
  return (
    <>
      <div className="display-xs">
        <MobileHeader />
      </div>

      <div className="display-md">
        <Header />
      </div>

      <Layout>
        <Row>
          <Col xs={3}>
            <FontAwesomeIcon icon={["fab", "github-square"]} />{" "}
          </Col>
        </Row>
      </Layout>
    </>
  );
};

export default CommunityMain;
