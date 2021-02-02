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
      metricsAndEvents: file(
        relativePath: { eq: "scope-metrics-and-events.mp4" }
      ) {
        publicURL
      }
      http: file(relativePath: { eq: "scope-http.mp4" }) {
        publicURL
      }
      dash: file(relativePath: { eq: "scope-dash.mp4" }) {
        publicURL
      }
    }
  `);

  // I am certain there is a better way of doing this but this will work.
  const images = [
    data.metricsAndEvents.publicURL,
    data.http.publicURL,
    data.dash.publicURL,
  ];

  const alt = ["Scope Metrics & Events", "Scope HTTP", "Scope Dashboard"];

  return (
    <Container className="highlights">
      <h2>{data.allHighlightsYaml.edges[0].node.title}</h2>
      {/*<p>{data.allHighlightsYaml.edges[0].node.description}</p> */}
      <Container>
        {data.allHighlightsYaml.edges[0].node.items.map((item, i) => {
          return (
            <Row>
              <Col
                xs={{ span: 12, order: 2 }}
                md={{ span: 6, order: item.order === 1 ? 2 : 1 }}
              >
                <video
                  autoPlay={true}
                  loop={true}
                  muted={true}
                  playsInline={true}
                  style={{ maxWidth: 90 + "%", margin: "10px auto" }}
                >
                  <source src={images[i]} type='video/mp4'>
                  </source>
                </video>
              </Col>
              <Col
                xs={{ span: 12, order: 1 }}
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
