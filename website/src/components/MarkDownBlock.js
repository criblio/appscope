import React from "react";
import { graphql } from "gatsby";
import Helmet from "react-helmet";
import Layout from "./layouts/documentationLayout";
import "prismjs/themes/prism.css";

export default function MarkDownBlock({ data }) {
  const post = data.markdownRemark;
  return (
    <>
      <Helmet title={post.frontmatter.title+ " | AppScope Docs"} />
      <Layout>
        <div
          className="code-container"
          dangerouslySetInnerHTML={{ __html: post.html }}
        />
      </Layout>
    </>
  );
}

export const query = graphql`
  query($slug: String!) {
    markdownRemark(fields: { slug: { eq: $slug } }) {
      html
      frontmatter {
        title
      }
    }
  }
`;
