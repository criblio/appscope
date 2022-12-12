import * as React from "react";
import Header from "../components/Header";
import MobileHeader from "../components/MobileHeader";
import Layout from "../components/layouts/community";
import { Row } from "react-bootstrap";
import Footer from "../components/Footer";
import "../utils/font-awesome";
import SEO from "../components/SEO";
import { Helmet } from "react-helmet";
import CommunityCard from "../components/widgets/CommunityCard";
import "../scss/_community.scss";

const CommunityMain = () => {
  return (
    <>
      <SEO />
      <Helmet>
        <meta name="og:image" content={'https://cribl.io/wp-content/uploads/2022/01/thumb.appScope.fullColorWhiteAlt.png'} />
      </Helmet>
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
