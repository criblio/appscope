import React, { useState } from "react";
import Helmet from "react-helmet";
import { useStaticQuery, graphql, Link } from "gatsby";
import "../scss/_docsNav.scss";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import Search from "./search";
import "../utils/font-awesome";

const searchIndices = [{ name: `Pages`, title: `Pages` }];
const DARK_MODE_KEY = 'DARK_MODE';

export default function DocsNav() {
  const [darkMode, toggleDarkMode] = useState(localStorage.getItem(DARK_MODE_KEY) === 'true');
  const [mobileNav, openMobileNav] = useState(false);
  const data = useStaticQuery(graphql`
    query DocumentationNav {
      allDocumentationNavYaml {
        nodes {
          name
          child {
            name
            path
          }
        }
      }
    }
  `);
  const navItems = data.allDocumentationNavYaml.nodes;
  const darkModeClick = () => {
    localStorage.setItem(DARK_MODE_KEY, !darkMode);
    toggleDarkMode(!darkMode);
  }

  return (
    <>
      <Helmet
        bodyAttributes={{
          class: darkMode ? "darkMode" : "",
        }}
      />
      <div className={mobileNav ? "docsNav mobileNavOpen" : "docsNav"}>
        <h2 className="mobileMenuHeader">Menu</h2>
        <Search indices={searchIndices} onChange={() => openMobileNav(true)} />

        <FontAwesomeIcon
          className="mobileDocsNav"
          icon={["fas", mobileNav ? "times" : "chevron-down"]}
          onClick={() => openMobileNav(mobileNav ? false : true)}
        />
        <h4>
          <FontAwesomeIcon
            icon={darkMode ? ["fas", "toggle-on"] : ["fas", "toggle-off"]}
            onClick={darkModeClick}
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
