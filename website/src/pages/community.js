import * as React from "react";
import Header from "../components/Header";
import MobileHeader from "../components/MobileHeader";
import Layout from "../components/layouts/community";
import { Row } from "react-bootstrap";
import Footer from "../components/Footer";
import "../utils/font-awesome";
import SEO from "../components/SEO";
import CommunityCard from "../components/widgets/CommunityCard";
import "../scss/_community.scss";

const CommunityMain = () => {
  return (
    <>
      <SEO />
      <div className="display-xs">
        <MobileHeader />
      </div>

      <div className="display-md">
        <Header />
      </div>

      <Layout>
        <Row>
          <CommunityCard />
        </Row>
      </Layout>
      <Footer />
    </>
  );
};

export default CommunityMain;
