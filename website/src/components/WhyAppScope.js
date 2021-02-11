import React from "react";
import { useStaticQuery, graphql } from "gatsby";
import { Container, Row, Col } from "react-bootstrap";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import "../scss/_whyAppscope.scss";
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
              icon
              title
            }
          }
        }
      }
    }
  `);

  return (
    <Container fluid className=" darkMode">
      <Container className="whyAppScope">
        <h2>{data.allWhyAppScopeYaml.edges[0].node.title}</h2>
        {/*<p>{data.allHighlightsYaml.edges[0].node.description}</p> */}
        <Container>
          <Row>
            {data.allWhyAppScopeYaml.edges[0].node.items.map((bullet, i) => {
              return (
                <Col xs={12} md={4} className="highlight-col" key={i}>
                  <h3>
                    <FontAwesomeIcon icon={bullet.icon} />
                  </h3>
                  <h4>{bullet.title}</h4>
                  <p>{bullet.item}</p>
                </Col>
              );
            })}
          </Row>
        </Container>
      </Container>
    </Container>
  );
}
