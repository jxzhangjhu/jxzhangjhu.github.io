 module Jekyll
  module HideCustomBibtex
    def hideCustomBibtex(input)
	  keywords = @context.registers[:site].config['filtered_bibtex_keywords']

	  keywords.each do |keyword|
<<<<<<< HEAD
		input = input.gsub(/^.*\b#{keyword}\b *= *\{.*$\n/, '')
=======
		input = input.gsub(/^.*#{keyword}.*$\n/, '')
>>>>>>> 320fd374 (Initial commit)
	  end

      return input
    end
  end
end

Liquid::Template.register_filter(Jekyll::HideCustomBibtex)
