import * as React from "react";
import Header from "../components/Header";
import NotFound from "../components/NotFound";

//import Alert from "../components/Alert";
import MobileHeader from "../components/MobileHeader";

const NotFoundPage = () => {
  return (
    <main>
      {/* <Alert /> */}
      <div className="display-xs">
        <MobileHeader />
      </div>

      <div className="display-md">
        <Header />
      </div>
      <NotFound />
    </main>
  );
};

export default NotFoundPage;
