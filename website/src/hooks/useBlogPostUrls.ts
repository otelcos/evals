import useBaseUrl from '@docusaurus/useBaseUrl';

interface BlogPostUrls {
  postUrl: string;
  imageUrl: string;
  avatarUrl: string;
}

interface BlogPostUrlParams {
  slug: string;
  image?: string;
  authorAvatar?: string;
}

export function useBlogPostUrls(params: BlogPostUrlParams): BlogPostUrls {
  const baseUrl = useBaseUrl('/');

  const postUrl = `${baseUrl}blog/${params.slug}`;

  const imageUrl = params.image
    ? `${baseUrl}${params.image.replace(/^\//, '')}`
    : `${baseUrl}img/blog/default-post.svg`;

  const avatarUrl = params.authorAvatar
    ? `${baseUrl}${params.authorAvatar.replace(/^\//, '')}`
    : `${baseUrl}img/authors/default-avatar.svg`;

  return { postUrl, imageUrl, avatarUrl };
}
