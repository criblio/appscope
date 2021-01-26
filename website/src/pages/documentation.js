import * as React from "react";
import Header from "../components/Header";
import Alert from "../components/Alert";
import MobileHeader from "../components/MobileHeader";
import Layout from "../components/layouts/documentationLayout";
import MarkDownBlock from "../components/MarkDownBlock";
import "../scss/_documentation.scss";
const DocumentationMain = () => {
  return (
    <>
      <div className="display-xs">
        <MobileHeader />
      </div>

      <div className="display-md">
        <Header />
      </div>

      <Layout>
        <div className="display-xs">
          <h1 style={{ padding: "0 10px" }}>Documentation</h1>
        </div>
        <MarkDownBlock />
      </Layout>
    </>
  );
};

export default DocumentationMain;
