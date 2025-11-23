require 'feedjira'
require 'httparty'
require 'jekyll'

module ExternalPosts
  class ExternalPostsGenerator < Jekyll::Generator
    safe true
    priority :high

    def generate(site)
      if site.config['external_sources'] != nil
        site.config['external_sources'].each do |src|
          begin
            p "Fetching external posts from #{src['name']}:"
            xml = HTTParty.get(src['rss_url']).body
            feed = Feedjira.parse(xml)
            feed.entries.each do |e|
              p "...fetching #{e.url}"
              slug = e.title.downcase.strip.gsub(' ', '-').gsub(/[^\w-]/, '')
              path = site.in_source_dir("_posts/#{slug}.md")
              doc = Jekyll::Document.new(
                path, { :site => site, :collection => site.collections['posts'] }
              )
              doc.data['external_source'] = src['name'];
              doc.data['feed_content'] = e.content;
              doc.data['title'] = "#{e.title}";
              doc.data['description'] = e.summary;
              doc.data['date'] = e.published;
              doc.data['redirect'] = e.url;
              site.collections['posts'].docs << doc
            end
          rescue => e
            p "Warning: Failed to fetch external posts from #{src['name']}: #{e.message}"
            Jekyll.logger.warn("ExternalPosts:", "Failed to fetch from #{src['name']}: #{e.message}")
          end
        end
      end
    end
  end

end
