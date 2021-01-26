import React from "react";
import { useStaticQuery, graphql } from "gatsby";

export default function MarkDownBlock() {
  const data = useStaticQuery(graphql`
    query codeQuery {
      markdownRemark(frontmatter: { title: { eq: "Intro" } }) {
        id
        html
      }
    }
  `);
  console.log(data);
  return (
    <div
      className="code-container"
      dangerouslySetInnerHTML={{
        __html: data.markdownRemark.html,
      }}
    ></div>
  );
}
