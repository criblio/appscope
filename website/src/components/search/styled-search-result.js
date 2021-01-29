import styled, { css } from "styled-components";
import SearchResult from "./search-result";

const Popover = css`
  max-height: 80vh;
  overflow: scroll;
  -webkit-overflow-scrolling: touch;
  position: absolute;
  z-index: 100000;
  top: 100%;
  margin-top: 0.5em;
  width: 80vw;
  max-width: 20em;
  box-shadow: 0 0 5px 0;
  padding: 1em;
  border-radius: 2px;
  background: #fff;
`;

export default styled(SearchResult)`
  display: ${(props) => (props.show ? `block` : `none`)};
  ${Popover}

  .HitCount {
    display: flex;
    justify-content: flex-end;
  }

  .Hits {
    ul {
      list-style: none;
      margin-left: 0;
    }

    li.ais-Hits-item {
      margin-bottom: 1em;
      text-transform: none;
      font-weight: 300;
      a {
        color: ${({ theme }) => theme.foreground};

        h4 {
          margin-bottom: 0.2em;
          span {
            font-size: 22px;
            font-weight: 300;
            margin-left: 0px;
            color: #fd6600;
          }
        }
      }
    }
  }

  .ais-PoweredBy {
    display: flex;
    justify-content: flex-end;
    font-size: 80%;

    svg {
      width: 70px;
    }
  }
`;
