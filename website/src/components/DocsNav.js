import React from "react";
import { useStaticQuery, graphql } from "gatsby";
import "../scss/_docsNav.scss";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";

export default function DocsNav() {
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
    <div className="docsNav">
      <input type="search" placeholder="Search..." />
      <h4>
        <FontAwesomeIcon icon={["fas", "toggle-off"]} /> <span>Dark</span>
      </h4>
      <ul>
        {navItems.map((item, i) => {
          return item.child !== null ? (
            <li key={i}>
              {item.name}
              <ul>
                {item.child.map((secondItem, j) => {
                  return secondItem.child !== null ? (
                    <li key={j}>{secondItem.name}</li>
                  ) : (
                    <li>{secondItem.name}</li>
                  );
                })}
              </ul>
            </li>
          ) : (
            <li key={i}>{item.name}</li>
          );
        })}
      </ul>
    </div>
  );
}
