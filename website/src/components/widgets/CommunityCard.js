import React from "react";
import { Link, useStaticQuery, navigate } from "gatsby";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import { Container, Row, Col } from "react-bootstrap";
import "../../utils/font-awesome";
import "../../scss/_community.scss";

export default function CommunityCard(props) {
  const data = useStaticQuery(graphql`
    query communityQuery {
      allCommunityYaml {
        edges {
          node {
            item
            link
            icon
            description
          }
        }
      }
    }
  `);
  console.log(data);
  return (
    <>
      {data.allCommunityYaml.edges.map((community, i) => {
        return (
          <Col xs={12} md={6} className="communityLink">
            <Link
              to={community.node.link}
              target="_black"
              rel="noopener noreferrer nofollow"
            >
              <Container className="communityCard">
                <Row>
                  <Col xs={4}>
                    <FontAwesomeIcon icon={["fab", community.node.icon]} />
                  </Col>
                  <Col xs={8}>
                    <h3>{community.node.item}</h3>
                    <p>{community.node.description}</p>
                  </Col>
                </Row>
              </Container>
            </Link>{" "}
          </Col>
        );
      })}
    </>
  );
}
