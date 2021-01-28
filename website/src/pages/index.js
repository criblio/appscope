import * as React from "react";
import Header from "../components/Header";
import Hero from "../components/Hero";
import Highlights from "../components/Highlights";
import TwoCol from "../components/TwoCol";
import Alert from "../components/Alert";
import MobileHeader from "../components/MobileHeader";
import Footer from "../components/Footer";
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
      <Footer />
    </main>
  );
};

export default IndexPage;
