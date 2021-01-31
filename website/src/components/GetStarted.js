import React from "react";
import { useStaticQuery, graphql, Link } from "gatsby";
import Img from "gatsby-image";
import { Container, Row, Col, Button } from "react-bootstrap";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import community from "../images/community.svg";
import "../utils/font-awesome";

export default function GetStarted() {
  const data = useStaticQuery(graphql`
    query getStartedQuery {
      allReadyGetStartedYaml {
        edges {
          node {
            title
            description
            items {
              icon
              url
              buttonText
            }
          }
        }
      }
    }
  `);

  return (
    <Container fluid style={{ border: "1px solid #00CCCC" }}>
      <Container className="howItWorks">
        <Row>
          <Col xs={12} md={6}>
            <img
              alt="Community"
              src={community}
              style={{ maxWidth: 90 + "%", margin: "10px auto" }}
            />
          </Col>
          <Col xs={12} md={6} className="text-left">
            <h2>{data.allReadyGetStartedYaml.edges[0].node.title}</h2>
            <p>{data.allReadyGetStartedYaml.edges[0].node.description}</p>
            <Row>
              <Col>
                {data.allReadyGetStartedYaml.edges[0].node.items.map(
                  (bullet, i) => {
                    return (
                      <Link to={bullet.url}>
                        <Button style={{ width: 250, marginRight: 30 }}>
                          <FontAwesomeIcon icon={["fab", bullet.icon]} />
                          {" " + bullet.buttonText}
                        </Button>
                      </Link>
                    );
                  }
                )}
              </Col>
            </Row>
          </Col>
        </Row>
      </Container>
    </Container>
  );
}
