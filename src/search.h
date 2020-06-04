#ifndef __SEARCH_H__
#define __SEARCH_H__


//
// This provides a fast way to do string searches through data that may
// or may not contain strings.  (Searching for a needle in a haystack...)
// The interface is a little like regcomp()/regexec()/regfree().
//    needleCreate() is like regcomp()
//    needleFind() is like regexec()
//    needleDestroy() is like regfree()
//
// There are a couple of important differences from the regex family of
// functions.  1) needleCreate expects a literal string as its input
// argument, not a regular expression.  The search is case sensitive.
// 2) needleFind() has a length argument that indicates the size of the
// buffer that should be searched.  needleFind() does is not stop
// searching if it sees NULL characters/bytes.  If a match is found, it
// returns the offset of the first match, otherwise it returns -1.
//

typedef struct _needle_t needle_t;

needle_t*     needleCreate(const char *);
void          needleDestroy(needle_t**);
int           needleLen(needle_t*);

int           needleFind(needle_t*, char *, int);

#endif // __SEARCH_H__
