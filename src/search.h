#ifndef __SEARCH_H__
#define __SEARCH_H__


//
// This provides a fast way to do string searches through data that may
// or may not contain strings.
// The interface is a little like regcomp()/regexec()/regfree().
//    searchComp() is like regcomp()
//    searchExec() is like regexec()
//    searchFree() is like regfree()
//
// There are a couple of important differences from the regex family of
// functions.  1) searchComp expects a literal string as its input
// argument, not a regular expression.  The search is case sensitive.
// 2) searchExec() has a length argument that indicates the size of the
// buffer that should be searched.  searchExec() does is not stop
// searching if it sees NULL characters/bytes.  If a match is found, it
// returns the offset of the first match, otherwise it returns -1.
//

typedef struct _search_t search_t;

search_t*     searchComp(const char *);
void          searchFree(search_t**);
int           searchLen(search_t*);

int           searchExec(search_t*, char *, int);

#endif // __SEARCH_H__
