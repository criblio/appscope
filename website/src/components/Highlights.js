import React from "react";
import { useStaticQuery, graphql } from "gatsby";
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
            }
          }
        }
      }
    }
  `);

  return (
    <Container className="highlights">
      <h2>{data.allHighlightsYaml.edges[0].node.title}</h2>
      <p>{data.allHighlightsYaml.edges[0].node.description}</p>
      <Row>
        {data.allHighlightsYaml.edges[0].node.items.map((item, i) => {
          return (
            <Col xs={12} md={6} className="highlight-col" key={i}>
              <h3>
                <FontAwesomeIcon icon={item.icon} />
              </h3>
              <h4>{item.title}</h4>
              <p>{item.shortDescription}</p>
            </Col>
          );
        })}
      </Row>
    </Container>
  );
}
