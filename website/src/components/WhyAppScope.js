import React from "react";
import { useStaticQuery, graphql } from "gatsby";
import { Container, Row, Col } from "react-bootstrap";
import "../scss/_highlights.scss";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import "../utils/font-awesome";

export default function WhyAppScope() {
  const data = useStaticQuery(graphql`
    query whyAppScopeQuery {
      allWhyAppScopeYaml {
        edges {
          node {
            title
            items {
              item
            }
          }
        }
      }
    }
  `);

  return (
    <Container className="highlights">
      <h2>{data.allWhyAppScopeYaml.edges[0].node.title}</h2>
      <Row>
        {data.allWhyAppScopeYaml.edges[0].node.items.map((bullet, i) => {
          return (
            <Col xs={12} md={12} className="highlight-col" key={i}>
              <p>{bullet.item}</p>
            </Col>
          );
        })}
      </Row>
    </Container>
  );
}
