import styled, { css } from "styled-components";
import SearchBox from "./search-box";

const open = css`
  background: ${({ theme }) => theme.background};
  cursor: text;
`;

const closed = css`
  background: transparent;
  cursor: pointer;
  padding-left: 1em;
`;

export default styled(SearchBox)`
  display: flex;
  flex-direction: row;
  align-items: center;
  margin-bottom: 0;

  .SearchInput {
    outline: none;
    border: 1px solid #ccc;
    font-size: 0.75em;
    transition: 100ms;
    border-radius: 2px;
    color: ${({ theme }) => theme.foreground};
    ::placeholder {
      color: ${({ theme }) => theme.faded};
    }
    ${({ hasFocus }) => (hasFocus ? open : closed)}
  }
`;
