import * as React from "react";
import Header from "../components/Header";
import Hero from "../components/Hero";
import Highlights from "../components/Highlights";
import TwoCol from "../components/TwoCol";
import Alert from "../components/Alert";
import MobileHeader from "../components/MobileHeader";
import Footer from "../components/Footer";
import WhyAppScope from "../components/WhyAppScope";
import HowItWorks from "../components/HowItWorks";
import GetStarted from "../components/GetStarted";

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
      <WhyAppScope />
      <HowItWorks />
      <GetStarted />
      <Footer />
    </main>
  );
};

export default IndexPage;
