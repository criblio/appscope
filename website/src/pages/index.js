import * as React from "react";
import Header from "../components/Header";
import Hero from "../components/Hero";
import Highlights from "../components/Highlights";
import TwoCol from "../components/TwoCol";
import Alert from "../components/Alert";
import MobileHeader from "../components/MobileHeader";

const IndexPage = () => {
  return (
    <main>
      {/* <Alert /> */}
      <div className="display-xs">
        <MobileHeader />
      </div>

      <div className="display-md">
        <Header />
      </div>
      <Hero />
      <Highlights />
      <TwoCol />
    </main>
  );
};

export default IndexPage;
