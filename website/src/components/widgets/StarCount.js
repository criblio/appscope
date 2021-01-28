import React, { useState, useEffect } from "react";
import { navigate } from "gatsby";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import "../../scss/_starCount.scss";
import "../../utils/font-awesome";

export default function StarCount() {
  const [starsCount, setStarsCount] = useState(0);

  useEffect(() => {
    fetch(`https://api.github.com/repos/criblio/appscope`)
      .then((response) => response.json())
      .then((resultData) => {
        setStarsCount(resultData.stargazers_count);
      });
  }, []);

  return (
    <div
      className="starCount-container"
      onClick={() => navigate("https://github.com/criblio/appscope")}
    >
      <div className="gitLogo">
        <FontAwesomeIcon icon={["fab", "github-square"]} />
      </div>
      <div className="starCount">
        <FontAwesomeIcon icon={"star"} />
        <span className="count">
          {starsCount > 999 ? (starsCount / 1000).toFixed(1) + "K" : starsCount}
        </span>
      </div>
    </div>
  );
}
