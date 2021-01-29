import React, { useState } from "react";
import Helmet from "react-helmet";

import { useStaticQuery, graphql, Link } from "gatsby";
import "../scss/_docsNav.scss";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";

export default function DocsNav() {
  const dm = document.querySelector("body").classList.contains("darkMode");
  const [darkMode, toggleDarkMode] = useState(false);
  const data = useStaticQuery(graphql`
    query DocumentationNav {
      allDocumentationNavYaml {
        nodes {
          name
          path
          child {
            name
            path
          }
        }
      }
    }
  `);
  const navItems = data.allDocumentationNavYaml.nodes;
  return (
    <>
      <Helmet
        bodyAttributes={{
          class: darkMode ? "darkMode" : "",
        }}
      />
      <div className="docsNav">
        <input type="search" placeholder="Search..." />
        <h4>
          <FontAwesomeIcon
            icon={darkMode ? ["fas", "toggle-on"] : ["fas", "toggle-off"]}
            onClick={() => {
              darkMode ? toggleDarkMode(false) : toggleDarkMode(true);
            }}
          />
          <span>{darkMode ? "Light" : "Dark"}</span>
        </h4>
        <ul>
          {navItems.map((item, i) => {
            return item.child !== null ? (
              <li key={i}>
                <Link
                  to={item.path}
                  activeStyle={{ color: "#FD6600", fontWeight: 700 }}
                >
                  {" "}
                  {item.name}
                </Link>
                <ul>
                  {item.child.map((secondItem, j) => {
                    return secondItem.child !== null ? (
                      <li key={j}>
                        <Link
                          to={secondItem.path}
                          activeStyle={{ color: "#FD6600", fontWeight: 700 }}
                        >
                          {secondItem.name}
                        </Link>
                      </li>
                    ) : (
                      ""
                    );
                  })}
                </ul>
              </li>
            ) : (
              <li key={i}>
                <Link to={item.path}>{item.name}</Link>
              </li>
            );
          })}
        </ul>
      </div>
    </>
  );
}
