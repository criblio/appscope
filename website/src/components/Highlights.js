import React from "react";
import { useStaticQuery, graphql } from "gatsby";
import Img from "gatsby-image";
import { Container, Row, Col } from "react-bootstrap";
import "../scss/_highlights.scss";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import "../utils/font-awesome";

export default function Highlights() {
  const data = useStaticQuery(graphql`
    query highlightsQuery {
      allHighlightsYaml {
        edges {
          node {
            title
            description
            items {
              title
              shortDescription
              icon
              order
            }
          }
        }
      }
      file(relativePath: { eq: "hero-image.png" }) {
        childImageSharp {
          fluid {
            sizes
            src
            srcSet
            base64
            aspectRatio
          }
        }
      }
    }
  `);

  return (
    <Container className="highlights">
      <h2>{data.allHighlightsYaml.edges[0].node.title}</h2>
      {/*<p>{data.allHighlightsYaml.edges[0].node.description}</p> */}
      <Container>
        {data.allHighlightsYaml.edges[0].node.items.map((item, i) => {
          return (
            <Row>
              <Col xs={12} md={{ span: 6, order: item.order === 1 ? 2 : 1 }}>
                <Img
                  fluid={data.file.childImageSharp.fluid}
                  alt="AppScope Machine"
                  style={{ maxWidth: 90 + "%", margin: "10px auto" }}
                />
              </Col>
              <Col
                xs={12}
                md={{ span: 6, order: item.order === 1 ? 1 : 2 }}
                className="text-left"
              >
                <h4>{item.title}</h4>
                <p>{item.shortDescription}</p>
              </Col>
            </Row>
          );
        })}
      </Container>
    </Container>
  );
}
